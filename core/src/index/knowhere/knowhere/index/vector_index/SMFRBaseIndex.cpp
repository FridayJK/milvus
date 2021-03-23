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

        SMFRRunTimeInfo runTimeInfo;
        SMFRIOMem       ioMem;

        index_smfr->SMFRGetRunTimeInfo(runTimeInfo);

        std::shared_ptr<uint8_t[]> dim_data(new uint8_t[sizeof(int)]);
        // std::shared_ptr<uint8_t[]> n_base(new uint8_t[sizeof(uint64_t)]);
        std::shared_ptr<uint8_t[]> index_data(new uint8_t[runTimeInfo.n_base]);

        memcpy(dim_data.get(), &runTimeInfo.n_dims, sizeof(int));
        // memcpy(n_base.get(), &runTimeInfo.n_base, sizeof(uint64_t));
        ioMem.data = index_data.get();
        index_smfr->SMFRIOSerialize(ioMem);

        int64_t index_length = ioMem.total*runTimeInfo.n_dims;
        //check ioMem.total==runTimeInfo.n_base

        BinarySet res_set;
        
        res_set.Append("smfr_dim", dim_data, sizeof(int));
        // res_set.Append("smfr_nbase", n_base, sizeof(uint64_t));
        res_set.Append("smfr_int8", index_data, index_length);

        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
SMFRBaseIndex::LoadImpl(const BinarySet& binary_set, const IndexType& type) {
    uint8_t         *index_data;
    uint64_t        index_length;
    int             dim;

    auto BinarySet_Dim = binary_set.GetByName("smfr_dim");
    auto BinarySet_index = binary_set.GetByName("smfr_int8");

    // dim = *(int)BinarySet_Dim->data.get();
    memcpy(&dim, BinarySet_Dim->data.get(), sizeof(int));
    index_length = BinarySet_index->size;
    index_data = BinarySet_index->data.get();

    SMFeatRetrieval *index = new SMFeatRetrieval(SM_BRUTE_FORCE_INT8, dim, SMFR_CPU);//
    index->SMFRLoadIndex((void*)index_data, dim, index_length);

    index_.reset(index);

    SealImpl();
}

}  // namespace knowhere
}  // namespace milvus
