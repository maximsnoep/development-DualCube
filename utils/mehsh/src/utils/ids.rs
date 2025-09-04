use bimap::BiHashMap;
use slotmap::DefaultKey;
use slotmap::SecondaryMap;
use slotmap::SlotMap;
use std::hash::Hash;
use std::marker::PhantomData;

#[derive(Default, Copy, Clone, Eq)]
pub struct Key<K, M> {
    raw: DefaultKey,
    _marker: PhantomData<(K, M)>,
}

impl<K, M> PartialEq for Key<K, M> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl<K, M> Hash for Key<K, M> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

impl<K, M> PartialOrd for Key<K, M> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.raw.cmp(&other.raw))
    }
}

impl<K, M> std::fmt::Debug for Key<K, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Idx({:?})", self.raw)
    }
}

impl<K, M> Key<K, M> {
    #[must_use]
    pub const fn new(raw: DefaultKey) -> Self {
        Self { raw, _marker: PhantomData }
    }

    #[must_use]
    pub const fn raw(self) -> DefaultKey {
        self.raw
    }
}

#[derive(Default, Clone, Debug)]
pub struct IdxMap<K, M, V> {
    map: SlotMap<DefaultKey, V>,
    _marker: PhantomData<(K, M)>,
}

impl<K, M, V> IdxMap<K, M, V> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: SlotMap::with_key(),
            _marker: PhantomData,
        }
    }

    pub fn insert(&mut self, value: V) -> Key<K, M> {
        Key::new(self.map.insert(value))
    }

    pub fn remove(&mut self, key: Key<K, M>) -> bool {
        self.map.remove(key.raw()).is_some()
    }

    #[must_use]
    pub fn get(&self, key: Key<K, M>) -> Option<&V> {
        self.map.get(key.raw())
    }

    #[must_use]
    pub fn get_mut(&mut self, key: Key<K, M>) -> Option<&mut V> {
        self.map.get_mut(key.raw())
    }

    #[must_use]
    pub fn contains(&self, key: Key<K, M>) -> bool {
        self.map.contains_key(key.raw())
    }

    pub fn ids(&self) -> impl Iterator<Item = Key<K, M>> + '_ {
        self.map.keys().map(|raw| Key::new(raw))
    }

    pub fn vals(&self) -> impl Iterator<Item = &V> + '_ {
        self.map.values()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[derive(Default, Clone, Debug)]
pub struct AssMap<K1, K2, M> {
    map: SecondaryMap<DefaultKey, DefaultKey>,
    _marker: PhantomData<(K1, K2, M)>,
}

impl<K1, K2, M> AssMap<K1, K2, M> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: SecondaryMap::new(),
            _marker: PhantomData,
        }
    }

    pub fn insert(&mut self, k1: Key<K1, M>, k2: Key<K2, M>) -> Option<Key<K2, M>> {
        if let Some(raw) = self.map.insert(k1.raw(), k2.raw()) {
            return Some(Key { raw, _marker: PhantomData });
        }
        None
    }

    #[must_use]
    pub fn get(&self, key: Key<K1, M>) -> Option<Key<K2, M>> {
        if let Some(&raw) = self.map.get(key.raw()) {
            return Some(Key { raw, _marker: PhantomData });
        }
        None
    }

    #[must_use]
    pub fn contains(&self, key: Key<K1, M>) -> bool {
        self.map.contains_key(key.raw())
    }

    pub fn ids(&self) -> impl Iterator<Item = Key<K1, M>> + '_ {
        self.map.keys().map(|raw| Key::new(raw))
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[derive(Default, Clone, Debug)]
pub struct IdMap<K, M>
where
    K: Eq + Hash,
    M: Eq + Hash,
{
    map: BiHashMap<usize, Key<K, M>>,
    _marker: PhantomData<(K, M)>,
}
impl<K, M> IdMap<K, M>
where
    K: Eq + Hash,
    M: Eq + Hash,
{
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: BiHashMap::new(),
            _marker: PhantomData,
        }
    }

    pub fn insert(&mut self, id: usize, key: Key<K, M>) {
        self.map.insert(id, key);
    }

    #[must_use]
    pub fn key(&self, id: usize) -> Option<&Key<K, M>> {
        self.map.get_by_left(&id)
    }

    #[must_use]
    pub fn id(&self, key: &Key<K, M>) -> Option<&usize> {
        self.map.get_by_right(key)
    }

    #[must_use]
    pub fn contains_id(&self, id: usize) -> bool {
        self.map.contains_left(&id)
    }

    #[must_use]
    pub fn contains_key(&self, key: &Key<K, M>) -> bool {
        self.map.contains_right(key)
    }
}

#[derive(Default, Clone, Debug)]
pub struct SecMap<K, M, V>
where
    K: Eq + Hash,
    M: Eq + Hash,
{
    map: SecondaryMap<DefaultKey, V>,
    _marker: PhantomData<(K, M)>,
}
impl<K, M, V> SecMap<K, M, V>
where
    K: Eq + Hash,
    M: Eq + Hash,
{
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: SecondaryMap::new(),
            _marker: PhantomData,
        }
    }

    pub fn insert(&mut self, key: &Key<K, M>, val: V) {
        self.map.insert(key.raw, val);
    }

    pub fn remove(&mut self, key: &Key<K, M>) -> Option<V> {
        self.map.remove(key.raw)
    }

    #[must_use]
    pub fn get(&self, key: &Key<K, M>) -> Option<&V> {
        self.map.get(key.raw)
    }

    #[must_use]
    pub fn get_or_panic(&self, key: Key<K, M>) -> &V {
        self.map.get(key.raw).unwrap()
    }

    #[must_use]
    pub fn get_mut(&mut self, key: &Key<K, M>) -> Option<&mut V> {
        self.map.get_mut(key.raw)
    }

    #[must_use]
    pub fn contains_key(&self, key: &Key<K, M>) -> bool {
        self.map.contains_key(key.raw)
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = (Key<K, M>, &V)> + '_ {
        self.map.iter().map(|(raw, val)| (Key::new(raw), val))
    }
}
