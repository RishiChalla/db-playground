use std::{fs::{File, OpenOptions}, io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write}, num::NonZeroUsize, path::Path};

use lru::LruCache;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use ciborium::{ser, de, from_reader, into_writer};

/// The size of a single page (data blob section in the B-Tree Index file)
const PAGE_SIZE: usize = 4000;

/// The size of the LRU cache, we hardcode this to take up 4gb of Heap RAM of pages.
const LRU_CACHE_SIZE: usize = 4_000_000_000 / PAGE_SIZE;

/// The offset number of pages, to reach a given page, this uniquely identified every tree node.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default, Clone, Copy)]
struct PageOffset(usize);

/// Uniquely identifies a record in the records database.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default, Clone, Copy)]
struct RecordRow(usize);

/// Represents a child of a tree node. This is a light-weight representation and doesn't contain the full data, simply a pointer to the page.
#[derive(Debug)]
enum TreeNodeChild {
    /// The child is a leaf node, with actual data.
    Leaf { record: RecordRow },
    /// The child is another branch.
    Branch { page: PageOffset },
    /// This child is uninitialized / empty. We use fix-width arrays to store the number of splits rather than vectors for
    /// compile-time optimizations and constant size pages. This results in a possibility of an empty tree node child.
    Empty,
}

/// Contains a node of the B+ Tree. This contains actual data, including splitting factors and/or leaves and record locations.
/// This is a dense data format, and is cached by LRU, and indexed by PageOffset.
#[derive(Debug)]
struct TreeNode<Key: PartialOrd, const MAX_SPLITS: usize> {
    /// The offset of the page, in the index file containing the B+ Tree
    offset: PageOffset,
    /// The depth of this particular tree node in the tree.
    depth: usize,
    /// A pointer to the left node, useful for traversing the B+ Tree sequentially.
    /// None if this is the left-most node of this depth.
    left: Option<PageOffset>,
    /// A pointer to the right node, useful for traversing the B+ Tree sequentially.
    /// None if this is the right-most node of this depth.
    right: Option<PageOffset>,
    /// A pointer to the parent node, useful for tarversing up the B+ Tree.
    /// None if this is the root.
    parent: Option<PageOffset>,
    /// The left-most child of this node.
    first: TreeNodeChild,
    /// All possible children of this node, up to MAX_SPLITS. Will always contain at least 0.5*MAX_SPLITS populated.
    splits: [(Key, TreeNodeChild); MAX_SPLITS],
}

/// A B+ Tree is used for indexing a Database of records by a particular key type. A single database may contain multiple indexed fields.
/// The Tree is held on disk, in an indexing file, and pages of it is read to RAM when needed on lookups. Lookups or insertions yield a relevant
/// Record Row, which both uniquely identifies database records, and provides random access.
#[derive(Debug)]
struct BPlusTree<Key: PartialOrd, const MAX_SPLITS: usize> {
    /// The cache containing node pages in memory. If a tree node is not present in this cache, it must be loaded from disk, and will replace
    /// the most unused node from this cache.
    node_cache: LruCache<PageOffset, TreeNode<Key, MAX_SPLITS>>,
    /// The handle of the index file, storing this B+ Tree. This file may be bigger than RAM, and will be read to/from with caching strategies.
    index_file: File,
    /// The inner B+ Tree data contains meta-data about the tree. This is always the first page of the index file.
    inner: BPlusTreeInner,
}

/// Contains the interior data of the B+ Tree. All interior data must be loaded and synchronized with the disk-saved B+ Tree.
#[derive(Debug, Default, Serialize, Deserialize)]
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

/// An error that may occur during Disk Operations with the B+ Tree.
#[derive(Debug)]
enum DiskError {
    /// There was an IO Stream issue.
    Io(io::Error),
    /// There was a serialization issue.
    Serialization(ser::Error<io::Error>),
    /// There was a deserialization issue.
    Deserialization(de::Error<io::Error>)
}

impl<Key: PartialOrd, const MAX_SPLITS: usize> BPlusTree<Key, MAX_SPLITS> {
    /// Loads the B+ Tree from an index file path.
    fn load(index_file_path: String) -> Result<Self, DiskError> {
        // Create a new LRU cache with the given cache size.
        let node_cache = LruCache::new(NonZeroUsize::new(LRU_CACHE_SIZE).unwrap());
        // If the file already exists, we're loading it, otherwise we're creating a new file.
        let exists = Path::new(&index_file_path).is_file();
        let index_file = OpenOptions::new()
            .read(true).write(true).create_new(!exists)
            .open(index_file_path)
            .map_err(DiskError::Io)?;
        let inner = BPlusTreeInner::default();
        // Create the tree and write / read the inner data.
        let mut tree = Self { node_cache, index_file, inner };
        if exists {
            tree.inner = read_page(&tree.index_file, PageOffset(0))?;
        } else {
            write_page(&tree.index_file, PageOffset(0), &tree.inner)?;
        }

        Ok(tree)
    }
}

/// Writes a single page of data into the B-Tree Index File.
fn write_page<Data: Serialize>(index_file: &File, page: PageOffset, data: &Data) -> Result<(), DiskError> {
    // Create a writer and set its position to the page
    let mut writer = BufWriter::new(index_file);
    writer.seek(SeekFrom::Start((page.0 * PAGE_SIZE) as u64)).map_err(DiskError::Io)?;
    // Create and populate a buffer with the serialized data, padded with 0s up to the page size.
    let mut buffer = Vec::with_capacity(PAGE_SIZE);
    into_writer(&data, &mut buffer).map_err(DiskError::Serialization)?;
    buffer.resize(PAGE_SIZE, 0);
    // Write the buffer to the writer, and flush it.
    writer.write_all(&buffer).map_err(DiskError::Io)?;
    writer.flush().map_err(DiskError::Io)?;
    Ok(())
}

/// Reads a single page of Data from the B-Tree index file and returns it.
fn read_page<Data: DeserializeOwned>(index_file: &File, page: PageOffset) -> Result<Data, DiskError> {
    // Create a reader and set its position to the page
    let mut reader = BufReader::new(index_file);
    reader.seek(SeekFrom::Start((page.0 * PAGE_SIZE) as u64)).map_err(DiskError::Io)?;
    // Read a page into the memory.
    let mut buffer = vec![0; PAGE_SIZE];
    reader.read_exact(&mut buffer).map_err(DiskError::Io)?;
    // Deserialize the page and save it.
    let data = from_reader(buffer.as_slice()).map_err(DiskError::Deserialization)?;
    Ok(data)
}
