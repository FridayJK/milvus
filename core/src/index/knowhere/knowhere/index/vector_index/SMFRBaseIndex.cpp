// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

// #include <faiss/index_io.h>
#include <fiu-local.h>

#include "knowhere/common/Exception.h"
#include "knowhere/index/vector_index/SMFRBaseIndex.h"
#include "knowhere/index/vector_index/IndexType.h"
#include "knowhere/index/vector_index/helpers/FaissIO.h"

namespace milvus {
namespace knowhere {

BinarySet
SMFRBaseIndex::SerializeImpl(const IndexType& type) {
    try {
        fiu_do_on("SMFRBaseIndex.SerializeImpl.throw_exception", throw std::exception());
        SMFeatRetrieval* index_smfr = index_.get();

        MemoryIOWriter writer;
        // faiss::write_index(index, &writer);
        std::shared_ptr<uint8_t[]> data(writer.data_);

        BinarySet res_set;
        // TODO(linxj): use virtual func Name() instead of raw string.
        res_set.Append("IVF", data, writer.rp);
        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
SMFRBaseIndex::LoadImpl(const BinarySet& binary_set, const IndexType& type) {
    auto binary = binary_set.GetByName("IVF");

    MemoryIOReader reader;
    reader.total = binary->size;
    reader.data_ = binary->data.get();

    // faiss::Index* index = faiss::read_index(&reader);
    // index_.reset(index);

    SealImpl();
}

}  // namespace knowhere
}  // namespace milvus
