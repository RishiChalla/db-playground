use super::*;

const MAX_SPLITS: usize = 10;

#[derive(Debug)]
struct TreeSplitter<Key: PartialOrd, Record: Clone> {
    key: Key,
    child: Box<TreeNode<Key, Record>>,
}

#[derive(Debug)]
enum TreeNode<Key: PartialOrd, Record: Clone> {
    Leaf {
        key: Key,
        record: Record,
    },
    Branch {
        first: Box<TreeNode<Key, Record>>,
        splits: [Option<TreeSplitter<Key, Record>>; MAX_SPLITS],
    },
}

#[derive(Debug)]
pub struct BPlusTree<Key: PartialOrd, Record: Clone> {
    root: Option<TreeNode<Key, Record>>,
}

impl<Key: PartialOrd, Record: Clone> BPlusTree<Key, Record> {

}

pub enum InsertionError {}
pub enum RemovalError {}
pub enum UpdateError {}
pub enum LookupError { InvalidLeaf, KeyNotFound }

impl<Key: PartialOrd, Record: Clone> Database<Key, Record> for BPlusTree<Key, Record> {
    fn new() -> Self {
        Self::default()
    }

    type InsertionError = InsertionError;
    fn insert_record(&mut self, key: Key, record: Record) -> Result<(), Self::InsertionError> {
        // Get the root, or insert the root if not available.
        let Some(mut node) = self.root.as_mut() else {
            self.root = Some(TreeNode::Leaf { key, record });
            return Ok(());
        };

        while let TreeNode::Branch { first, splits } = node {
            // Find partition point in the splitting branch
            let idx = splits.partition_point(|split| {
                let Some(split) = split else { return false };
                key > split.key
            });

            // Update the node, based on the partition point
            node = {
                let (left, right) = splits.split_at_mut(idx);
                // First branch is a special case
                if idx == 0 { first }
                // Partition found in the splits
                else if let Some(split) = right[0].as_mut() { &mut split.child }
                // Partition is at the last populated branch.
                else if let Some(split) = left[idx - 1].as_mut() { &mut split.child }
                // The last branch wasn't populated - this is a corrupt node. We panic since the integrity of the database has been violated.
                else { panic!("Corrupt branch found."); }
            };
        }

        Ok(())
    }

    type RemovalError = RemovalError;
    fn remove_record(&mut self, key: Key) -> Result<(), Self::RemovalError> {
        todo!()
    }

    type UpdateError = UpdateError;
    fn update_record(&mut self, key: Key, record: Record) -> Result<(), Self::InsertionError> {
        todo!()
    }

    type LookupError = LookupError;
    fn lookup_record(&self, key: Key) -> Result<Record, Self::LookupError> {
        let Some(mut node) = self.root.as_ref() else { return Err(LookupError::KeyNotFound) };

        while let TreeNode::Branch { first, splits } = node {
            // Find partition point in the splitting branch
            let idx = splits.partition_point(|split| {
                let Some(split) = split else { return false };
                key > split.key
            });

            // Update the node, based on the partition point
            node = {
                // First branch is a special case
                if idx == 0 { first }
                // Partition found in the splits
                else if let Some(split) = splits[idx].as_ref() { &split.child }
                // Partition is at the last populated branch.
                else if let Some(split) = splits[idx - 1].as_ref() { &split.child }
                // The last branch wasn't populated - this is a corrupt node. We panic since the integrity of the database has been violated.
                else { panic!("Corrupt branch found."); }
            };
        }

        // We found a leaf node (the while loop should not end without a leaf, so the error should be impossible)
        let TreeNode::Leaf { key: found_key, record } = node else { return Err(LookupError::InvalidLeaf) };
        // Check if the leaf node is the correct key, if not then we did not find the key.
        if *found_key == key { Ok(record.clone()) }
        else { Err(LookupError::KeyNotFound) }
    }
}

impl<Key: PartialOrd, Record: Clone> Default for BPlusTree<Key, Record> {
    fn default() -> Self {
        Self { root: Default::default() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup() {
    }

}
