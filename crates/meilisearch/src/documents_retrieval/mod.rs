use index_scheduler::IndexScheduler;
use meilisearch_auth::AuthFilter;
use meilisearch_types::milli::progress::Progress;
use uuid::Uuid;

use crate::search::{Federation, SearchQueryWithIndex};

pub struct DocumentsRetrieval {
    index_scheduler: IndexScheduler,
    queries: Vec<SearchQueryWithIndex>,
    federation: Option<Federation>,
    auth: Option<AuthFilter>,
    is_proxy: bool,
    include_metadata: bool,
    request_uid: Uuid,
}

impl DocumentsRetrieval {
    pub fn new(
        index_scheduler: IndexScheduler,
        request_uid: Uuid,
        queries: Vec<SearchQueryWithIndex>,
    ) -> Self {
        Self {
            index_scheduler,
            queries,
            federation: None,
            is_proxy: false,
            include_metadata: false,
            request_uid,
            auth: None,
        }
    }

    pub fn with_federation(&mut self, federation: Federation) -> &mut Self {
        self.federation = Some(federation);
        self
    }

    pub fn with_auth(&mut self, auth: AuthFilter) -> &mut Self {
        self.auth = Some(auth);
        self
    }

    pub fn is_proxy(&mut self, is_proxy: bool) -> &mut Self {
        self.is_proxy = is_proxy;
        self
    }

    pub fn include_metadata(&mut self, include_metadata: bool) -> &mut Self {
        self.include_metadata = include_metadata;
        self
    }

    pub fn execute(&self, progress: &Progress) -> () {}
}
