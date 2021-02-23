#ifndef __SM_FEAT_RETRIEVAL__
#define __SM_FEAT_RETRIEVAL__
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>

#ifdef __cplusplus
extern "C"{
#endif

//SMFR索引类型
typedef enum __SMFR_WORK_TYPES__{
    SM_BRUTE_FORCE_INT8,        //int8暴力检索
    SM_LEVEL2,
    SM_LEVEL3,
}SMFR_INDEX_TYPES;

//
typedef enum __SMFR_WORK_MODE__{
    SMFR_CPU = 0,
    SMFR_GPU,
}SMFR_DEVICE_MODE;

//特征版本信息
typedef struct __SMFRVersionInfo__{
    int version;
    int date;
}SMFRVersionInfo;

//配置信息
typedef struct __SMFRConfig__{
    int  distance_type; //计算距离类型L2、cos
    int  n_probe;       //查询分片个数
    int  n_thread;      //线程数
}SMFRConfig;
//配置类型
typedef enum __SMFRConfigType__{
    DISTANCE_TYPE = 0,
    PROBE_NUM,
    THREAD_NUM,
}SMFRConfigType;

//运行时信息
typedef struct __SMFRRunTimeInfo__{
    SMFR_DEVICE_MODE  device_type;      //设备类型
    SMFR_INDEX_TYPES  index_type;       //索引类型
    std::string       index_key;        //索引字符配置
    int               n_base;           //底库数量
    int               n_dims;           //特征维度
    int               distance_type;    //距离类型
    int               n_probe = 10;     //查询分片个数
    int               n_thread;         //omp线程数
}SMFRRunTimeInfo;

//索引建立、检索、修改、删除，动态info
class SMFeatRetrieval{
    public:
    explicit SMFeatRetrieval(SMFR_INDEX_TYPES types, int featDims, SMFR_DEVICE_MODE deviceMode);//
    explicit SMFeatRetrieval(std::string oriIndexPath, SMFR_DEVICE_MODE deviceMode);//
    ~SMFeatRetrieval();

    /*---------------底库数据导入-----------------
    baseFeat         -I       底库数据
    ---------------------------------*/
    int SMFRAddDataBase(std::vector <float> &baseFeat);

    /*--------------特征检索------------------
    queryFeat           -I      待检索特征
    topK                -I      取最相近的topk个
    res                 -O      输出结果<id,distance>对
    ---------------------------------*/
    int SMFRSearch(std::vector <float> &queryFeat, int topK, std::vector<std::pair<int64_t,float>> &res);

    /*单条特征修改或预删除(置0)
    singleFeat          -I      单条特征向量
    id                  -I      向量ID
    */
    int SMFRUpdateSingleID(std::vector <float> &singleFeat,int64_t id);

    /*删除已被置0的数据*/
    int SMFRRemoveIDs();

    // int SMFRTrainIndex(const std::vector <float> &trainData);
    // int SMFRSaveIndex(std::string saveIndexPath);
    // int SMFRLoadIndex(std::string loadIndexPath);
    // int SMFRAddWithIDs();
    // int SMFRRemoveWithIDs();

    //获取版本信息
    int SMFRGetVersionInfo(SMFRVersionInfo &versionInfo);

    //获取运行时信息
    int SMFRGetRunTimeInfo(SMFRRunTimeInfo &runTimeInfo);

    //设置运行时信息
    int SMFRSetRunTimeInfo(SMFRConfig &configInfo, SMFRConfigType configType);
    
    private:
    SMFRVersionInfo      modelVersionInfo;//
    SMFRRunTimeInfo      runTimeInfo_;
    void                 *smfr_handlde;
};

class SMFRFeatUpdate{
    SMFRFeatUpdate();//校验版本信息，
    ~SMFRFeatUpdate();

    //获取模型版本信息
    int SMFRGetVersionInfo(SMFRVersionInfo &versionInfo);

    int SMFRCheckVersion(const SMFRVersionInfo &v_old,const SMFRVersionInfo &v_new);
    int SMFRFeatUpdateOP(const std::vector <float> &data_old,std::vector <float> &data_new);//数据量大时可分批载入升级
};

#ifdef __cplusplus
}
#endif

#endif
