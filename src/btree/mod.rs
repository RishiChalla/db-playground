use std::{cell::RefCell, fs::{File, OpenOptions}, io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write}, num::NonZeroUsize, path::Path};

use lru::LruCache;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use ciborium::{ser, de, from_reader, into_writer};

/// The size of a single page (data blob section in the B-Tree Index file)
const PAGE_SIZE: usize = 4000;

/// The size of the LRU cache, we hardcode this to take up 4gb of Heap RAM of pages.
const LRU_CACHE_SIZE: usize = 4_000_000_000 / PAGE_SIZE;

/// The offset number of pages, to reach a given page, this uniquely identified every tree node.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Clone, Copy)]
pub struct PageOffset(usize);

impl Default for PageOffset {
    /// Default is 1 instead of 0 since first page is reserved for meta-data
    fn default() -> Self { Self(1) }
}

/// Uniquely identifies a record in the records database.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default, Clone, Copy)]
struct RecordRow(usize);

/// The variant of a tree node, a tree node can either be a branch or a leaf.
/// This changes the interpretation of the usize in `TreeNode::splits` and `TreeNode::first` to a RecordRow for leaves,
/// and a PageOffset for branches.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
enum TreeNodeVariant {
    /// The child is a leaf node, with actual data.
    Leaf,
    /// The child is another branch.
    Branch,
}

/// Contains a node of the B+ Tree. This contains actual data, including splitting factors and/or leaves and record locations.
/// This is a dense data format, and is cached by LRU, and indexed by PageOffset.
#[derive(Debug, Serialize, Deserialize, Clone)]
struct TreeNode<Key: Ord + Serialize + Clone> {
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

impl<Key: Ord + Serialize + DeserializeOwned + Clone> Default for TreeNode<Key> {
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

impl<Key: Ord + Serialize + DeserializeOwned + Clone> TreeNode<Key> {
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

        // Check if the key already exists.
        if self.splits[partition - 1].0 == *key {
            return Err(IndexingError::InsertionAlreadyExists);
        }

        // Insert at the partition point to retain sorted order.
        self.splits.insert(partition, (key.clone(), child));

        // The new key was smaller than everything in the node. This means we need to traverse upwards and update all parent pointers.
        if partition == 0 {
            let mut parent = self.parent;
            while let Some(offset) = parent {
                let mut node = offset.load_node(tree)?;
                node.splits[0].0 = key.clone();
                parent = node.parent;
                mutated_nodes.push(node); // Mark the parent as mutated.
            }
        }

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
struct BPlusTree<Key: Ord + Serialize + DeserializeOwned + Clone> {
    /// The cache containing node pages in memory. If a tree node is not present in this cache, it must be loaded from disk, and will replace
    /// the most unused node from this cache.
    node_cache: RefCell<LruCache<PageOffset, TreeNode<Key>>>,
    /// The handle of the index file, storing this B+ Tree. This file may be bigger than RAM, and will be read to/from with caching strategies.
    index_file: File,
    /// The inner B+ Tree data contains meta-data about the tree. This is always the first page of the index file.
    inner: BPlusTreeInner,
}

/// Contains the interior data of the B+ Tree. All interior data must be loaded and synchronized with the disk-saved B+ Tree.
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
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
    Serialization(ser::Error<io::Error>),
    /// There was a deserialization issue.
    /// > The Disk and RAM state is guaranteed to be stable, even if this error occurs.
    Deserialization(de::Error<io::Error>),
    /// Occurs when a Tree node is attempted to be written into the dedicated inner metadata block.
    /// > The Disk and RAM state is guaranteed to be stable, even if this error occurs.
    MetadataOverwrite,
    /// Occurs when attempting to insert onto a key that already exists.
    InsertionAlreadyExists,
}

impl<Key: Ord + Serialize + DeserializeOwned + Clone> BPlusTree<Key> {
    /// Loads the B+ Tree from an index file path.
    fn load(index_file_path: String) -> Result<Self, IndexingError> {
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
            node.offset.write_page(&mut self.index_file, &node)
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

    /// Writes a single page of data into the B-Tree Index File.
    fn write_page<Data: Serialize>(&self, index_file: &mut File, data: &Data) -> Result<(), IndexingError> {
        // Create a writer and set its position to the page
        let mut writer = BufWriter::new(&mut *index_file);
        writer.seek(SeekFrom::Start((self.0 * PAGE_SIZE) as u64)).map_err(IndexingError::Io)?;
        // Create and populate a buffer with the serialized data, padded with 0s up to the page size.
        let mut buffer = Vec::with_capacity(PAGE_SIZE);
        into_writer(&data, &mut buffer).map_err(IndexingError::Serialization)?;
        buffer.resize(PAGE_SIZE, 0);
        // Write the buffer to the writer, and flush it.
        writer.write_all(&buffer).map_err(IndexingError::Io)?;
        writer.flush().map_err(IndexingError::Io)?;
        drop(writer);
        index_file.sync_data().map_err(IndexingError::Io)?;
        Ok(())
    }
    
    /// Reads a single page of Data from the B-Tree index file and returns it.
    fn read_page<Data: DeserializeOwned>(&self, index_file: &File) -> Result<Data, IndexingError> {
        // Create a reader and set its position to the page
        let mut reader = BufReader::new(index_file);
        reader.seek(SeekFrom::Start((self.0 * PAGE_SIZE) as u64)).map_err(IndexingError::Io)?;
        // Read a page into the memory.
        let mut buffer = vec![0; PAGE_SIZE];
        reader.read_exact(&mut buffer).map_err(IndexingError::Io)?;
        // Deserialize the page and save it.
        let data = from_reader(buffer.as_slice()).map_err(IndexingError::Deserialization)?;
        Ok(data)
    }

    /// Attempts to load a node, using the LRU Caching strategy. If the node is not present in cache, it is retrieved from disk, and cached for future use.
    fn load_node<Key>(&self, tree: &BPlusTree<Key>) -> Result<TreeNode<Key>, IndexingError>
    where Key: Ord + Serialize + DeserializeOwned + Clone {
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
    use super::{BPlusTree, RecordRow};

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

    /// Tests that the B+ Tree data is persisted in the DB file.
    #[test]
    fn test_btree_persistence() {
        reset_db_file("test_database").expect("Existing DB file must be cleared");

        // Create empty B-Tree (ensure it is empty)
        let mut btree: BPlusTree<i32> = BPlusTree::load(String::from("test-data/test_database")).unwrap();
        assert_eq!(btree.lookup(&10).unwrap(), None);
        btree.insert(&10, RecordRow(3)).unwrap();
        assert_eq!(btree.lookup(&10).unwrap(), Some(RecordRow(3)));

        // Drop it From Memory
        drop(btree);

        // Load existing B-Tree and assert our previously inserted key is persisted.
        let btree: BPlusTree<i32> = BPlusTree::load(String::from("test-data/test_database")).unwrap();
        assert_eq!(btree.lookup(&10).unwrap(), Some(RecordRow(3)));
    }

}
