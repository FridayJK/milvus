#pragma once

#include <memory>
#include <vector>

#include "knowhere/common/Exception.h"
#include "knowhere/common/Log.h"
#include "knowhere/index/vector_index/SMFRBaseIndex.h"
#include "knowhere/index/vector_index/VecIndex.h"
#include "SMFeatRetrieval.hpp"

namespace milvus {
namespace knowhere {

class SMFRInt8 : public VecIndex, public SMFRBaseIndex {
 public:
    SMFRInt8() : SMFRBaseIndex(nullptr) {
        index_type_ = IndexEnum::INDEX_SMFR_INT8;
    }

    explicit SMFRInt8(std::shared_ptr<SMFeatRetrieval> index) : SMFRBaseIndex(std::move(index)) {
        index_type_ = IndexEnum::INDEX_SMFR_INT8;
    }

    BinarySet
    Serialize(const Config& config = Config()) override;

    void
    Load(const BinarySet&) override;

    // void
    // BuildAll(const DatasetPtr&, const Config&) override;

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

    // int64_t
    // IndexSize() override {
    //     return Count() * Dim() * sizeof(BinaryType);
    // }

    // void
    // UpdateIndexSize() override;

    // VecIndexPtr
    // CopyCpuToGpu(const int64_t, const Config&);

    // virtual const float*
    // GetRawVectors();

 private:
    virtual void
    QueryImpl(int64_t, int64_t, const float*, int64_t, float*, int64_t*, const Config&);
    // int64_t gpu_;
    // std::shared_ptr<SMFeatRetrieval> index_;
};

using SMFRInt8Ptr = std::shared_ptr<SMFRInt8>;

}  // namespace knowhere
}  // namespace milvus
