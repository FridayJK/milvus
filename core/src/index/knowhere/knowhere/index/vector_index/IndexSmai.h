#pragma once

#include <memory>
#include <vector>

#include "knowhere/common/Exception.h"
#include "knowhere/common/Log.h"
#include "knowhere/index/vector_index/VecIndex.h"
#include "knowhere/index/vector_index/SMFeatRetrieval.hpp"

namespace milvus {
namespace knowhere {

class SmaiIndex : public VecIndex {
 public:
    explicit SmaiIndex(const int64_t gpu_num = -1) : gpu_(gpu_num) {
        if (gpu_ >= 0) {
            index_mode_ = IndexMode::MODE_GPU;
        }
        index_type_ = IndexEnum::INDEX_NSG;
    }

    BinarySet
    Serialize(const Config& config = Config()) override;

    void
    Load(const BinarySet&) override;

    void
    BuildAll(const DatasetPtr&, const Config&) override;

    void
    Train(const DatasetPtr&, const Config&) override {
        KNOWHERE_THROW_MSG("Smai not support build item dynamically, please invoke BuildAll interface.");
    }

    void
    AddWithoutIds(const DatasetPtr&, const Config&) override;

    DatasetPtr
    Query(const DatasetPtr&, const Config&) override;

    int64_t
    Count() override;

    int64_t
    Dim() override;

    void
    UpdateIndexSize() override;

 private:
    int64_t gpu_;
    std::shared_ptr<SMFeatRetrieval> index_;
};

using SmaiIndexPtr = std::shared_ptr<SmaiIndex>();

}  // namespace knowhere
}  // namespace milvus
