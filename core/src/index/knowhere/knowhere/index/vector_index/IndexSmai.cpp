
#include <fiu-local.h>
#include <string>

#include "knowhere/common/Exception.h"
#include "knowhere/common/Timer.h"
#include "knowhere/index/vector_index/IndexIVF.h"
#include "knowhere/index/vector_index/IndexType.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/SMFeatRetrieval.hpp"

#ifdef MILVUS_GPU_VERSION
#include "knowhere/index/vector_index/helpers/Cloner.h"
#endif

namespace milvus {
namespace knowhere {

BinarySet
SmaiIndex::Serialize(const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
}

void
SmaiIndex::Load(const BinarySet& index_binary) {
}

DatasetPtr
SmaiIndex::Query(const DatasetPtr& dataset_ptr, const Config& config) {
    return null;
}

void
SmaiIndex::BuildAll(const DatasetPtr& dataset_ptr, const Config& config) {
}

int64_t
SmaiIndex::Count() {
    return 0
}

int64_t
SmaiIndex::Dim() {
    return 0;
}

void
SmaiIndex::UpdateIndexSize() {

}

}  // namespace knowhere
}  // namespace milvus
