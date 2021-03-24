
#include <fiu-local.h>
#include <string>

#include "knowhere/common/Exception.h"
#include "knowhere/index/vector_index/IndexSMFR.h"
#include "knowhere/common/Timer.h"
#include "knowhere/index/vector_index/IndexIVF.h"
#include "knowhere/index/vector_index/IndexType.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"

#ifdef MILVUS_GPU_VERSION
#include "knowhere/index/vector_index/helpers/Cloner.h"
#endif

namespace milvus {
namespace knowhere {

BinarySet
SMFRInt8::Serialize(const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
    return SerializeImpl(index_type_);
}

void
SMFRInt8::Load(const BinarySet& binary_set) {
    LoadImpl(binary_set, index_type_);
}

void
SMFRInt8::AddWithoutIds(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }

    GETTENSOR(dataset_ptr)
    std::vector<float> tmpFeat(dim*rows);
    memcpy(tmpFeat.data(), p_data, sizeof(float)*dim*rows);
    // index_->add(rows, (float*)p_data);
    index_->SMFRAddDataBase(tmpFeat);
}

DatasetPtr
SMFRInt8::Query(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    GETTENSOR(dataset_ptr)

    int64_t k = config[meta::TOPK].get<int64_t>();
    auto elems = rows * k;
    size_t p_id_size = sizeof(int64_t) * elems;
    size_t p_dist_size = sizeof(float) * elems;
    auto p_id = (int64_t*)malloc(p_id_size);
    auto p_dist = (float*)malloc(p_dist_size);

    QueryImpl(dim, rows, (float*)p_data, k, p_dist, p_id, config);
    MapOffsetToUid(p_id, static_cast<size_t>(elems));

    auto ret_ds = std::make_shared<Dataset>();
    ret_ds->Set(meta::IDS, p_id);
    ret_ds->Set(meta::DISTANCE, p_dist);
    return ret_ds;
}

// void
// SMFRInt8::BuildAll(const DatasetPtr& dataset_ptr, const Config& config) {
// }

int64_t
SMFRInt8::Count() {
    SMFRRunTimeInfo runTimeInfo;
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    index_->SMFRGetRunTimeInfo(runTimeInfo);
    return runTimeInfo.n_base;
}

int64_t
SMFRInt8::Dim() {
    SMFRRunTimeInfo runTimeInfo;
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    index_->SMFRGetRunTimeInfo(runTimeInfo);
    return runTimeInfo.n_dims;
}

// void
// SMFRInt8::UpdateIndexSize() {
// }

void
SMFRInt8::QueryImpl(int64_t d, int64_t n, const float* data, int64_t k, float* distances, int64_t* labels, const Config& config) {
    std::vector <float> queryfeats(n*d);
    std::vector<std::pair<int64_t,float>> pair_res(n*k);
    memcpy(queryfeats.data(), data, sizeof(float)*d*n);
    index_->SMFRSearch(queryfeats, k, pair_res);
    for(int i=0; i<n; i++){
        distances[i] = pair_res[i].second;
        labels[i]    = pair_res[i].first;
    }
}

}  // namespace knowhere
}  // namespace milvus
