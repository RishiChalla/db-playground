mod b_plus_tree_db;

pub trait Database<Key, Record> {
    fn new() -> Self;

    type InsertionError;
    fn insert_record(&mut self, key: Key, record: Record) -> Result<(), Self::InsertionError>;
    
    type RemovalError;
    fn remove_record(&mut self, key: Key) -> Result<(), Self::RemovalError>;

    type UpdateError;
    fn update_record(&mut self, key: Key, record: Record) -> Result<(), Self::InsertionError>;

    type LookupError;
    fn lookup_record(&self, key: Key) -> Result<Record, Self::LookupError>;
}
