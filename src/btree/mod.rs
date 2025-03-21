use std::{
    cell::RefCell, fmt::Debug, fs::{File, OpenOptions}, io::{self, BufReader, BufWriter, Seek, SeekFrom, Write}, num::NonZeroUsize, ops::Deref, path::Path
};

use bincode::{
    config::{self, Configuration, LittleEndian, Fixint, Limit},
    decode_from_reader,
    encode_to_vec,
    error::{DecodeError, EncodeError},
    Decode,
    Encode,
    de::{read::Reader, Decoder},
    enc::{write::Writer, Encoder},
};
use lru::LruCache;
use arrayvec::ArrayString;

/// A marker trait for an Indexable Key that can be used as a key for the B+ Tree Index.
trait IndexKey: Debug + Ord + Encode + Decode<()> + Clone {}
impl<Key: Debug + Ord + Encode + Decode<()> + Clone> IndexKey for Key {}

/// A String can be used as a key for indexing the B+ Tree, but it must have a max-length.
/// This uses ArrayString from the arrayvec crate to provide a viable string key.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
struct StringKey<const MAX_SIZE: usize>(ArrayString::<MAX_SIZE>);

impl<const MAX_SIZE: usize> Deref for StringKey<MAX_SIZE> {
    type Target = ArrayString<MAX_SIZE>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, const MAX_SIZE: usize> TryFrom<&'a str> for StringKey<MAX_SIZE> {
    type Error = <&'a str as TryInto<ArrayString<MAX_SIZE>>>::Error;
    
    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        Ok(Self(ArrayString::try_from(value)?))
    }
}

impl<const MAX_SIZE: usize> Encode for StringKey<MAX_SIZE> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        (self.0.len() as u32).encode(encoder)?;
        encoder.writer().write(self.0.as_bytes())
    }
}

impl<const MAX_SIZE: usize> Decode<()> for StringKey<MAX_SIZE> {
    fn decode<D: Decoder<Context = ()>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let len = u32::decode(decoder)?;
        let mut bytes: [u8; MAX_SIZE] = [0; MAX_SIZE];
        decoder.reader().read(&mut bytes)?;
        let mut string = ArrayString::from_byte_string(&bytes).map_err(|inner| DecodeError::Utf8 { inner })?;
        string.truncate(len as usize);
        Ok(Self(string))
    }
}

/// The size of a single page (data blob section in the B-Tree Index file)
const PAGE_SIZE: usize = 2 << 11;

/// The size of the LRU cache, we hardcode this to take up 4mb of Heap RAM of pages.
const LRU_CACHE_SIZE: usize = (2 << 21) / PAGE_SIZE;

/// The offset number of pages, to reach a given page, this uniquely identified every tree node.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Encode, Decode, Clone, Copy)]
pub struct PageOffset(usize);

impl Default for PageOffset {
    /// Default is 1 instead of 0 since first page is reserved for meta-data
    fn default() -> Self { Self(1) }
}

/// Uniquely identifies a record in the records database.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Encode, Decode, Default, Clone, Copy)]
struct RecordRow(usize);

/// The variant of a tree node, a tree node can either be a branch or a leaf.
/// This changes the interpretation of the usize in `TreeNode::splits` and `TreeNode::first` to a RecordRow for leaves,
/// and a PageOffset for branches.
#[derive(Debug, Encode, Decode, Clone, Copy, PartialEq, Eq)]
enum TreeNodeVariant {
    /// The child is a leaf node, with actual data.
    Leaf,
    /// The child is another branch.
    Branch,
}

/// Contains a node of the B+ Tree. This contains actual data, including splitting factors and/or leaves and record locations.
/// This is a dense data format, and is cached by LRU, and indexed by PageOffset.
#[derive(Debug, Encode, Decode, Clone, PartialEq, Eq)]
struct TreeNode<Key: IndexKey> {
    /// The offset of the page, in the index file containing the B+ Tree
    offset: PageOffset,
    /// A pointer to the left node, useful for traversing the B+ Tree sequentially.
    /// None if this is the left-most node of this depth.
    left: Option<PageOffset>,
    /// A pointer to the right node, useful for traversing the B+ Tree sequentially.
    /// None if this is the right-most node of this depth.
    right: Option<PageOffset>,
    /// A pointer to the parent node, useful for tarversing up the B+ Tree.
    /// None if this is the root.
    parent: Option<PageOffset>,
    /// All possible children of this node, up to a MAX length determined by the Key size, relative to the page size.
    /// Will always contain at least 0.5*MAX_SPLITS populated.
    /// All items in a child of a split, will always be greater than or equal to its associated key.
    /// We opt to use a vector instead of fixed-arrays since Serde does not support const generics.
    /// The child is a RecordRow when this is a leaf node, or a PageOffset when this is a branch node.
    splits: Vec<(Key, usize)>,
    /// Whether this is a leaf or branch node, this determines the interpretation of `splits` and `first`.
    variant: TreeNodeVariant,
}

impl<Key: IndexKey> Default for TreeNode<Key> {
    fn default() -> Self {
        Self {
            offset: PageOffset(0),
            left: None,
            right: None,
            parent: None,
            splits: Vec::new(),
            variant: TreeNodeVariant::Leaf,
        }
    }
}

impl<Key: IndexKey> TreeNode<Key> {
    /// Retrieves the maximum number of splits in a single tree node.
    /// This is determined by the size of the Keys, relative to the size of a page.
    const fn max_splits() -> usize {
        (PAGE_SIZE - std::mem::size_of::<Self>()) / std::mem::size_of::<(Key, usize)>()
    }

    /// Uses binary search to find the first item after 
    fn partition_point(&self, key: &Key) -> usize {
        self.splits.partition_point(|(split, _child)| *key >= *split)
    }

    /// Inserts an item into this tree node, and recurses up if necessary.
    /// All items are only modified in memory, and placed into the mutated_nodes vector. The disk is not written to at any point,
    /// nor is the tree cache in RAM written to.
    fn insert_item(mut self, key: &Key, child: usize, tree: &BPlusTree<Key>, inner: &mut BPlusTreeInner, mutated_nodes: &mut Vec<Self>)
        -> Result<(), IndexingError> {
        // Use binary search to find the partition point
        let partition = self.partition_point(key);

        // The new key was smaller than everything in the node. This means we need to traverse upwards and update all parent pointers.
        // If partition is 0, we're also guaranteed the key doesn't exist, since it was smaller than everything we already have.
        if partition == 0 {
            let mut parent = self.parent;
            while let Some(offset) = parent {
                let mut node = offset.load_node(tree)?;
                node.splits[0].0 = key.clone();
                parent = node.parent;
                mutated_nodes.push(node); // Mark the parent as mutated.
            }
        } else if self.splits[partition - 1].0 == *key { // Check if the key already exists.
            return Err(IndexingError::InsertionAlreadyExists);
        }

        // Insert at the partition point to retain sorted order.
        self.splits.insert(partition, (key.clone(), child));

        // The node has no space left, we need to split the node apart.
        if self.splits.len() > Self::max_splits() {
            // Split the node, where the original node has the left-side. The right side is placed into a new node.
            let split_partition = self.splits.len() / 2;
            let right = self.splits.split_off(split_partition);
            let mut right_node: TreeNode<Key> = TreeNode {
                offset: BPlusTree::<Key>::get_insertion_point(inner),
                left: Some(self.offset),
                right: self.right,
                parent: self.parent,
                splits: right,
                variant: self.variant,
            };
            self.right = Some(right_node.offset);

            // We need to add the right side to the parent.
            if let Some(parent) = self.parent {
                let parent = parent.load_node(tree)?;
                parent.insert_item(&right_node.splits[0].0, right_node.offset.0, tree, inner, mutated_nodes)?;
            } else { // We have no parent, we need to make a new root and increase the depth of the tree.
                inner.depth += 1;
                let root = TreeNode {
                    offset: BPlusTree::<Key>::get_insertion_point(inner),
                    splits: vec![(self.splits[0].0.clone(), self.offset.0), (right_node.splits[0].0.clone(), right_node.offset.0)],
                    variant: TreeNodeVariant::Branch,
                    ..Default::default()
                };
                inner.root = Some(root.offset);
                self.parent = Some(root.offset);
                right_node.parent = Some(root.offset);
                mutated_nodes.push(root);
            }

            mutated_nodes.push(right_node);
        }

        // Mark the node as mutated.
        mutated_nodes.push(self);

        Ok(())
    }
}

/// A B+ Tree is used for indexing a Database of records by a particular key type. A single database may contain multiple indexed fields.
/// The Tree is held on disk, in an indexing file, and pages of it is read to RAM when needed on lookups. Lookups or insertions yield a relevant
/// Record Row, which both uniquely identifies database records, and provides random access.
#[derive(Debug)]
struct BPlusTree<Key: IndexKey> {
    /// The cache containing node pages in memory. If a tree node is not present in this cache, it must be loaded from disk, and will replace
    /// the most unused node from this cache.
    node_cache: RefCell<LruCache<PageOffset, TreeNode<Key>>>,
    /// The handle of the index file, storing this B+ Tree. This file may be bigger than RAM, and will be read to/from with caching strategies.
    index_file: File,
    /// The inner B+ Tree data contains meta-data about the tree. This is always the first page of the index file.
    inner: BPlusTreeInner,
}

/// Contains the interior data of the B+ Tree. All interior data must be loaded and synchronized with the disk-saved B+ Tree.
#[derive(Debug, Default, Encode, Decode, Clone, PartialEq, Eq)]
struct BPlusTreeInner {
    /// The root node of the B+ Tree
    root: Option<PageOffset>,
    /// The number of leaves in the B+ Tree (and thus records in the database)
    size: usize,
    /// The number of layers to the B+ Tree
    depth: usize,
    /// The offset to the next page insertion in the B+ Tree file.
    /// This is never decremented, instead when a page is removed, it is added to the free list to be re-used.
    next_page: PageOffset,
    /// A list of pages who've been removed and are available to be re-used. Insertions only create new pages after the free list is empty.
    free_list: Vec<PageOffset>,
}

/// An error that may occur during Indexing Operations with the B+ Tree.
#[derive(Debug)]
pub enum IndexingError {
    /// There was an IO Stream issue.
    /// > The Disk and RAM state is guaranteed to be stable, even if this error occurs.
    Io(io::Error),
    /// Upon attempting to restore the Index file state on a disk error, another disk error was encountered.
    /// > **Encountering this kind of error may leave the affected page(s) corrupted.**
    Reversion { pages: Vec<PageOffset> },
    /// There was a serialization issue.
    /// > The Disk and RAM state is guaranteed to be stable, even if this error occurs.
    Serialization(EncodeError),
    /// There was a deserialization issue.
    /// > The Disk and RAM state is guaranteed to be stable, even if this error occurs.
    Deserialization(DecodeError),
    /// Occurs when a Tree node is attempted to be written into the dedicated inner metadata block.
    /// > The Disk and RAM state is guaranteed to be stable, even if this error occurs.
    MetadataOverwrite,
    /// Occurs when attempting to insert onto a key that already exists.
    InsertionAlreadyExists,
    /// Occurs when attempting to write data larger than a page.
    WritePageOverflow,
}

impl<Key: IndexKey> BPlusTree<Key> {
    /// Loads the B+ Tree from an index file path.
    fn load<Str: Into<String>>(index_file_path: Str) -> Result<Self, IndexingError> {
        let index_file_path: String = index_file_path.into();
        // Create a new LRU cache with the given cache size.
        let node_cache = RefCell::new(LruCache::new(NonZeroUsize::new(LRU_CACHE_SIZE).unwrap()));
        // If the file already exists, we're loading it, otherwise we're creating a new file.
        let exists = Path::new(&index_file_path).is_file();
        let index_file = OpenOptions::new()
            .read(true).write(true).create_new(!exists)
            .open(index_file_path)
            .map_err(IndexingError::Io)?;
        let inner = BPlusTreeInner::default();
        // Create the tree and write / read the inner data.
        let mut tree = Self { node_cache, index_file, inner };
        if exists {
            tree.inner = PageOffset::inner().read_page(&tree.index_file)?;
        } else {
            PageOffset::inner().write_page(&mut tree.index_file, &tree.inner)?;
        }

        Ok(tree)
    }

    /// Gets the offset of the next insertion. Works on a copy of the inner data to support atomic insertions.
    fn get_insertion_point(inner: &mut BPlusTreeInner) -> PageOffset {
        if let Some(offset) = inner.free_list.pop() { // Remove and return the page from the free list if available
            offset
        } else { // Increment the next page counter and return it.
            let next_page = inner.next_page;
            inner.next_page.0 += 1;
            next_page
        }
    }

    /// Commits a series of transactions - that is updating both the inner and all provided nodes in an atomic operation.
    /// This commit either fully succeeds, or fully fails
    /// (in which case the state of the tree on Disk, and RAM is the same as prior to the transaction)
    fn commit_transaction(&mut self, inner: BPlusTreeInner, nodes: Vec<TreeNode<Key>>) -> Result<(), IndexingError> {
        // First we take a backup of the inner metadata before doing any writes.
        let inner_checkpoint = self.inner.clone();
        // Attempt the write
        let inner_write = PageOffset::inner().write_page(&mut self.index_file, &inner);
        if let Err(error) = inner_write { // Something went wrong, attempt to revert it.
            PageOffset::inner().write_page(&mut self.index_file, &inner_checkpoint)
                .map_err(|_| IndexingError::Reversion { pages: vec![PageOffset::inner()] })?; // Reversion failed.
            return Err(error);
        }

        // We save each node prior to the mutation, to revert to in the case of a failure.
        let mut node_checkpoints = Vec::with_capacity(nodes.len());
        let node_write = nodes.iter().try_for_each(|node| {
            // We only make checkpoints of overwrites. New insertions do not have data they are overwriting.
            if node.offset < inner_checkpoint.next_page {
                node_checkpoints.push(node.offset.load_node(self)?);
            }
            node.offset.write_page(&mut self.index_file, node)
        });

        // A write was unsuccessful, we need to revert all previously written pages.
        if let Err(error) = node_write {
            let mut reversion_errors = Vec::new(); // Gather a list of pages who we're unable to revert.
            // Attempt to revert the inner metadata
            if PageOffset::inner().write_page(&mut self.index_file, &inner_checkpoint).is_err() {
                reversion_errors.push(PageOffset::inner());
            }
            // Extend reversion errors by all nodes we're unable to revert.
            reversion_errors.extend(node_checkpoints.iter().filter_map(|node| {
                node.offset.write_page::<TreeNode<Key>>(&mut self.index_file, node).map_err(|_| node.offset).err()
            }));
            // We found reversion errors
            if !reversion_errors.is_empty() {
                Err(IndexingError::Reversion { pages: reversion_errors })
            } else { // Reversion was successful
                Err(error)
            }
        } else { // All writes were successful - we need to update the nodes and inner in RAM
            self.inner = inner;
            let mut node_cache = self.node_cache.borrow_mut();
            for node in nodes { node_cache.put(node.offset, node); }
            Ok(())
        }
    }

    /// Traverses the tree until it finds a leaf node where either the key resides as an exact match,
    /// or is matched by the leaf node's splitting criteria. Returns None if the tree is empty, and there is no root.
    fn traverse_tree(&self, key: &Key) -> Result<Option<TreeNode<Key>>, IndexingError> {
        // Get the root node location
        let Some(node) = self.inner.root else { return Ok(None) };
        // Start with the root node, keep traversing until we encounter a leaf node.
        let mut node = node.load_node(self)?;
        while node.variant == TreeNodeVariant::Branch {
            // Use a binary search to find the partition point, the number of splitters in a single node
            // will likely be very large on average for most key sizes.
            let partition = node.splits.partition_point(|(split, _child)| *key >= *split);
            // Fetch the next page offset based on the partition point.
            let split_idx = if partition == 0 { 0 } else { partition - 1 };
            // Traverse to the next node.
            node = PageOffset(node.splits[split_idx].1).load_node(self)?;
        }
        
        Ok(Some(node))
    }

    /// Inserts a key and its associated record location into the index tree.
    fn insert(&mut self, key: &Key, record: RecordRow) -> Result<(), IndexingError> {
        // We only modify a copy of the inner, to make operations atomic.
        let mut inner = self.inner.clone();
        let mut mutated_nodes = Vec::new();

        inner.size += 1;

        // There is a leaf present within the range for the given key, we can add our key into it.
        // Alternatively if our new key is smaller than every key in the database, we receive the left-most leaf.
        if let Some(leaf) = self.traverse_tree(key)? {
            // Insert the item into the leaf (which recursively inserts up if/when necessary)
            leaf.insert_item(key, record.0, self, &mut inner, &mut mutated_nodes)?;
            // Insert item internally uses recursion, which means new pages are actually in an inverse order
            // (ie new insertion at page 5 is before page 4). However attempting to write to page 5 before page 4 exists will cause
            // a write error, thus we reverse the mutations to ensure they're done in insertion order.
            // (We could also do a sort for more safety, but a reverse is faster and fine in this case)
            mutated_nodes.reverse();
        } else { // Create a new root at the insertion point.
            let root = TreeNode {
                offset: Self::get_insertion_point(&mut inner),
                splits: vec![(key.clone(), record.0)],
                variant: TreeNodeVariant::Leaf,
                ..Default::default()
            };
            inner.depth += 1;
            inner.root = Some(root.offset);
            mutated_nodes.push(root);
        }

        // No mutations are done up to this point, all mutations are done at once with this operation.
        self.commit_transaction(inner, mutated_nodes)
    }

    /// Removes a key and its associated record from the index tree.
    fn remove(&mut self, key: &Key) -> Result<(), IndexingError> {
        todo!()
    }

    /// Searches for the location of a record given its key. None is returned if there is no record found for
    /// the given key.
    fn lookup(&self, key: &Key) -> Result<Option<RecordRow>, IndexingError> {
        if let Some(leaf) = self.traverse_tree(key)? { // Traverse the tree to find a leaf
            // Binary search for the key within the leaf
            let record_idx = leaf.splits.binary_search_by_key(&key, |(split, _child)| split);
            if let Ok(record_idx) = record_idx { // Record found - return it
                Ok(Some(RecordRow(leaf.splits[record_idx].1)))
            } else { // No record found in the leaf.
                Ok(None)
            }
        } else { // No leaf found - the tree is empty
            Ok(None)
        }
    }
}

impl PageOffset {

    /// Gets the hardcoded page offset for the inner Tree metadata.
    const fn inner() -> Self { Self(0) }

    /// Config used for serialization to binary with bincode.
    const fn bincode_config() -> Configuration<LittleEndian, Fixint, Limit<PAGE_SIZE>> {
        config::standard()
            .with_fixed_int_encoding()
            .with_limit::<PAGE_SIZE>()
    }

    /// Writes a single page of data into the B-Tree Index File.
    fn write_page<Data: Encode + Decode<()> + Eq + Debug>(&self, index_file: &mut File, data: &Data) -> Result<(), IndexingError> {
        // Create a writer and set its position to the page
        let mut writer = BufWriter::new(&mut *index_file);
        writer.seek(SeekFrom::Start((self.0 * PAGE_SIZE) as u64)).map_err(IndexingError::Io)?;
        // Create and populate a buffer with the serialized data, padded with 0s up to the page size.
        let mut buffer = encode_to_vec(data, Self::bincode_config()).map_err(IndexingError::Serialization)?;
        if buffer.len() > PAGE_SIZE { return Err(IndexingError::WritePageOverflow); }
        buffer.resize(PAGE_SIZE, 0);
        // Write the buffer to the writer, and flush it.
        writer.write_all(&buffer).map_err(IndexingError::Io)?;
        writer.flush().map_err(IndexingError::Io)?;
        drop(writer);
        index_file.sync_data().map_err(IndexingError::Io)?;

        // Debug Only - Assert the page we just wrote to is actually equal to the data we wrote, when read.
        // TODO - Get rid of this (this is just for testing string keys.)
        // debug_assert_eq!(*data, self.read_page(index_file).unwrap());

        Ok(())
    }
    
    /// Reads a single page of Data from the B-Tree index file and returns it.
    fn read_page<Data: Decode<()>>(&self, index_file: &File) -> Result<Data, IndexingError> {
        // Create a reader and set its position to the page
        let mut reader = BufReader::new(index_file);
        reader.seek(SeekFrom::Start((self.0 * PAGE_SIZE) as u64)).map_err(IndexingError::Io)?;
        // Deserialize the page and save it.
        let data = decode_from_reader(reader, Self::bincode_config()).map_err(IndexingError::Deserialization)?;
        Ok(data)
    }

    /// Attempts to load a node, using the LRU Caching strategy. If the node is not present in cache, it is retrieved from disk, and cached for future use.
    fn load_node<Key: IndexKey>(&self, tree: &BPlusTree<Key>) -> Result<TreeNode<Key>, IndexingError> {
        if *self == Self::inner() { return Err(IndexingError::MetadataOverwrite); }
        let mut node_cache = tree.node_cache.borrow_mut();
        if let Some(node) = node_cache.get(self) {
            Ok(node.clone())
        } else {
            let node = self.read_page::<TreeNode<Key>>(&tree.index_file)?;
            node_cache.put(*self, node.clone());
            Ok(node)
        }
    }
}

#[cfg(test)]
mod test {
    use std::{fs::{create_dir, remove_file}, io::{self, ErrorKind}, path::Path};
    use bincode::{decode_from_slice, encode_to_vec};
    use super::{BPlusTree, PageOffset, RecordRow, StringKey, TreeNode, IndexingError};

    /// Resets the DB file, and creates a folder if necessary.
    fn reset_db_file(path: &'static str) -> Result<(), io::Error> {
        // Create the test data folder, ignore already existing error.
        if let Err(error) = create_dir("test-data/") {
            if error.kind() != ErrorKind::AlreadyExists { return Err(error); }
        }

        // Remove the DB file, ignore if the DB file doesn't already exist
        if let Err(error) = remove_file(Path::new(&(String::from("test-data/") + path))) {
            if error.kind() != ErrorKind::NotFound { return Err(error); }
        }

        Ok(())
    }

    /// Tests that the B+ Tree correctly handles overwriting attempts.
    #[test]
    fn test_btree_overwrite() {
        reset_db_file("overwrite").expect("Existing DB file must be cleared");

        let mut btree: BPlusTree<usize> = BPlusTree::load("test-data/overwrite").unwrap();
        btree.insert(&10, RecordRow(10)).unwrap();
        assert!(btree.insert(&10, RecordRow(20)).is_err_and(|err| matches!(err, IndexingError::InsertionAlreadyExists)));
        btree.insert(&5, RecordRow(20)).unwrap();
        assert!(btree.insert(&5, RecordRow(15)).is_err_and(|err| matches!(err, IndexingError::InsertionAlreadyExists)));
        assert!(btree.insert(&10, RecordRow(40)).is_err_and(|err| matches!(err, IndexingError::InsertionAlreadyExists)));
        btree.insert(&7, RecordRow(30)).unwrap();
    }

    /// Tests that the B+ Tree data splits correctly when inserting.
    #[test]
    fn test_btree_insertion_split() {
        reset_db_file("insertion-split").expect("Existing DB file must be cleared");

        // Create an empty B+ Tree and query its max splits per node.
        // max_splits ^ 2 will be guaranteed to have at least 2 levels, and will most likely have 3.
        let mut btree: BPlusTree<usize> = BPlusTree::load("test-data/insertion-split").unwrap();
        let num_items = TreeNode::<usize>::max_splits() * TreeNode::<usize>::max_splits();

        println!("Testing {} insertions.", num_items);

        // Insert all items into the B+ Tree.
        (0..num_items).try_for_each(|record| {
            btree.insert(&record, RecordRow(record))
        }).expect("Should be able to insert all keys.");

        // Assert that all added items are present
        (0..num_items).for_each(|record| {
            assert_eq!(btree.lookup(&record).unwrap(), Some(RecordRow(record)));
        });

        // Drop it From Memory
        drop(btree);

        // Load existing B-Tree and assert our previously inserted keys are persisted.
        let btree: BPlusTree<usize> = BPlusTree::load("test-data/insertion-split").unwrap();
        (0..num_items).for_each(|record| {
            assert_eq!(btree.lookup(&record).unwrap(), Some(RecordRow(record)));
        });
    }

    /// Tests that the string key serialization and deserialization is working properly.
    #[test]
    fn test_string_key() {
        let string = "abcd1234%^&*)\u{2122}";
        let key = StringKey::<256>::try_from(string).unwrap();
        let encoded = encode_to_vec(key.clone(), PageOffset::bincode_config()).unwrap();
        let decoded: StringKey::<256> = decode_from_slice(encoded.as_slice(), PageOffset::bincode_config()).unwrap().0;
        assert_eq!(key, decoded);
    }

    /// Tests that the B+ Tree string key works
    #[test]
    fn test_btree_string_key() {
        reset_db_file("string-key").expect("Existing DB file must be cleared");

        type EmailKey = StringKey::<256>; // Emails typically have a max length of 256.
        let mut btree: BPlusTree<EmailKey> = BPlusTree::load("test-data/string-key").unwrap();
        // Dummy emails from https://www.akto.io/tools/email-generator
        let emails = [ // 20 emails guarantees at least 1 split, since at 4kb page sizes, each page can only have up to 15 keys.
            "Austin59@gmail.com", "Orlo.Schulist73@gmail.com", "Leilani_Heller1@gmail.com",
            "Ambrose_Hayes46@gmail.com", "Valentina67@gmail.com", "Lemuel19@gmail.com",
            "Gerry_OConner12@gmail.com", "Hermina_Bogan@gmail.com", "Janessa58@gmail.com",
            "Makenzie.McDermott@gmail.com", "Lane16@gmail.com", "Julie.Flatley@gmail.com",
            "Deonte.Hermann61@gmail.com", "Archibald_Kutch@gmail.com", "Gonzalo_Rowe77@gmail.com",
            "Filiberto79@gmail.com", "Melyssa_Windler@gmail.com", "Nakia.Satterfield@gmail.com",
            "Buford.Bode68@gmail.com", "Marlen.Bruen@gmail.com",
        ];
        // Insert all the emails into the B+ Tree.
        emails.iter().enumerate().try_for_each(|(record, &email)| {
            btree.insert(&EmailKey::try_from(email).unwrap(), RecordRow(record))
        }).expect("Should be able to insert all keys.");

        // Assert that all emails were correctly stored, and can be looked up.
        emails.iter().enumerate().for_each(|(record, &email)| {
            assert_eq!(btree.lookup(&EmailKey::try_from(email).unwrap()).unwrap(), Some(RecordRow(record)));
        });

        // Drop it From Memory
        drop(btree);

        // Ensure that all insertions and the split were correctly persisted.
        let btree: BPlusTree<EmailKey> = BPlusTree::load("test-data/string-key").unwrap();
        emails.iter().enumerate().for_each(|(record, &email)| {
            assert_eq!(btree.lookup(&EmailKey::try_from(email).unwrap()).unwrap(), Some(RecordRow(record)));
        });
    }

}
