use index_scheduler::IndexScheduler;
use meilisearch_auth::AuthFilter;
use meilisearch_types::{
    error::ResponseError,
    milli::{progress::Progress, Deadline},
};
use uuid::Uuid;

use crate::{
    error::MeilisearchHttpError,
    extractors::authentication::AuthenticationError,
    search::{
        add_search_rules, perform_federated_search, FederatedSearchResult, Federation,
        SearchQueryWithIndex, SearchResultWithIndex,
    },
};

pub struct DocumentsRetrieval {
    queries: Vec<SearchQueryWithIndex>,
    federation: Option<Federation>,
    is_proxy: bool,
    include_metadata: bool,
    request_uid: Uuid,
}

impl DocumentsRetrieval {
    pub fn new(request_uid: Uuid, queries: Vec<SearchQueryWithIndex>) -> Self {
        Self { queries, federation: None, is_proxy: false, include_metadata: false, request_uid }
    }

    pub fn with_federation(&mut self, federation: Federation) -> &mut Self {
        self.federation = Some(federation);
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

    pub async fn execute(
        mut self,
        index_scheduler: actix_web::web::Data<IndexScheduler>,
        progress: &Progress,
    ) -> Result<DocumentsRetrievalResult, (ResponseError, Option<usize>)> {
        let features = index_scheduler.features();
        // Federated search
        if let Some(federation) = self.federation.take() {
            let (search_result, _) = perform_federated_search(
                index_scheduler,
                self.queries,
                federation,
                features,
                self.is_proxy,
                self.request_uid,
                self.include_metadata,
                progress,
            )
            .await
            .with_index(None)?;

            return Ok(DocumentsRetrievalResult::Federated(search_result));
        }

        // Multi-search
        let search_results: Result<_, (ResponseError, _)> = async {
            let mut search_results = Vec::with_capacity(self.queries.len());
            for (query_index, query) in self.queries.iter().enumerate() {
                if query.federation_options.is_some() {
                    return Err((
                        MeilisearchHttpError::FederationOptionsInNonFederatedRequest(query_index)
                            .into(),
                        Some(query_index),
                    ));
                }

                let (fixed_query, federation) = fixup_query_federation(query);

                let (search_result, _) = perform_federated_search(
                    index_scheduler.clone(),
                    vec![fixed_query],
                    federation,
                    features,
                    self.is_proxy,
                    self.request_uid,
                    self.include_metadata,
                    progress,
                )
                .await
                .with_index(Some(query_index))?;

                search_results.push(SearchResultWithIndex {
                    index_uid: query.index_uid.to_string(),
                    result: search_result.into_search_result(
                        query.q.clone().unwrap_or_default(),
                        query.index_uid.as_str(),
                    ),
                });
            }

            Ok(search_results)
        }
        .await;

        search_results.map(DocumentsRetrievalResult::Multi)
    }
}

fn fixup_query_federation(query: &SearchQueryWithIndex) -> (SearchQueryWithIndex, Federation) {
    let mut query = query.clone();
    // Move query parameters that make sense at the federation level
    // from the `SearchQueryWithIndex` to the `Federation`
    let SearchQueryWithIndex {
        index_uid,
        q: _,
        vector: _,
        media: _,
        hybrid: _,
        offset,
        limit,
        page,
        hits_per_page,
        attributes_to_retrieve: _,
        retrieve_vectors: _,
        attributes_to_crop: _,
        crop_length: _,
        attributes_to_highlight: _,
        show_ranking_score: _,
        show_ranking_score_details: _,
        show_performance_details,
        use_network: _,
        show_matches_position: _,
        filter: _,
        sort: _,
        distinct,
        facets,
        highlight_pre_tag: _,
        highlight_post_tag: _,
        crop_marker: _,
        matching_strategy: _,
        attributes_to_search_on: _,
        ranking_score_threshold: _,
        locales: _,
        personalize: _,
        federation_options: _,
    } = &mut query;

    let mut federation = Federation::default();
    let Federation {
        limit: federation_limit,
        offset: federation_offset,
        page: federation_page,
        hits_per_page: federation_hits_per_page,
        facets_by_index: _,
        merge_facets: _,
        show_performance_details: federation_show_performance_details,
        distinct: federation_distinct,
    } = &mut federation;

    if let Some(limit) = limit.take() {
        *federation_limit = limit;
    }
    if let Some(offset) = offset.take() {
        *federation_offset = offset;
    }
    if let Some(page) = page.take() {
        *federation_page = Some(page);
    }
    if let Some(hits_per_page) = hits_per_page.take() {
        *federation_hits_per_page = Some(hits_per_page);
    }
    if let Some(distinct) = distinct.take() {
        *federation_distinct = Some(distinct);
    }

    if let Some(show_performance_details) = show_performance_details.take() {
        *federation_show_performance_details = show_performance_details;
    }

    'facets: {
        if let Some(facets) = facets.take() {
            if facets.is_empty() {
                break 'facets;
            }
            let facets_by_index = federation.facets_by_index.entry(index_uid.clone()).or_default();
            *facets_by_index = Some(facets);
        }
    }

    (query, federation)
}

/// Local `Result` extension trait to avoid `map_err` boilerplate.
trait WithIndex {
    type T;
    /// convert the error type inside of the `Result` to a `ResponseError`, and
    /// return a couple of it + the usize.
    fn with_index(self, index: Option<usize>) -> Result<Self::T, (ResponseError, Option<usize>)>;
}

impl<T, E: Into<ResponseError>> WithIndex for Result<T, E> {
    type T = T;
    fn with_index(self, index: Option<usize>) -> Result<T, (ResponseError, Option<usize>)> {
        self.map_err(|err| (err.into(), index))
    }
}

pub enum DocumentsRetrievalResult {
    Federated(FederatedSearchResult),
    Multi(Vec<SearchResultWithIndex>),
}
