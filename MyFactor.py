from typing import *
import pandas as pd
import numpy as np
import datetime as dt
import pymysql
import openpyxl
from time import time
from tqdm import tqdm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from linearmodels import FamaMacBeth
plt.style.use('ggplot')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from itertools import product
from scipy.stats import mode
from sqlalchemy import create_engine
from sqlalchemy.types import VARCHAR,DECIMAL


class MyFactor:
    def __init__(self,host:str,user:str,password:str,exchange:str,loadtrd:str = None,start:str = None,end:str = None):
        '''
        连接因子数据库,获取因子表列表,交易日历

        参数:

        host,user,password:数据库地址,账号,密码

        exchange:交易所简称,目前数据库中包含'sz','sh'两个交易所的交易日历

        loadtrd:读取的交易数据类型,例如{'stk','fund'}(时间较久),一般做回测时需要读取,简单存取因子值无需读取,loadtrd为None时不读取

        start:读取交易数据的开始日期,'%Y%m%d'格式的日期,start为None时从头开始读

        end:读取交易数据的结束日期,'%Y%m%d'格式的日期,end为None时读取到最后一个日期

        (不建议start和end设为None,读取数据负担会很大)
        '''
        self.host,self.user,self.password,self.assettype = host,user,password,loadtrd
        self.factortabledict = self.getfactortables()
        self.trdcalendar = self.gettrdcalendar(exchange)
        self.df_trd = self.getassettrd(['adjclose','size'],start,end) if loadtrd is not None else None

    @staticmethod
    def cumret(ret:Union[list,np.array,pd.Series])->float:
        '''根据连续收益率序列计算累计收益率'''
        return np.exp(sum(np.log(1 + np.array(ret)))) - 1

    @staticmethod
    def ret2annual(df_ret:pd.DataFrame)->float:
        '''
        根据时间序列收益率计算年化收益率
        
        df_ret:共2列
            第1列为日期,values为str,格式为'%Y%m%d'
            第2列为区间收益率,values为float,从当前日期到下一个日期期间的收益率
        '''
        df_ret.columns = ['trddate','ret']
        df_ret = df_ret.sort_values('trddate')
        start,end = df_ret['trddate'].values[0],df_ret['trddate'].values[-1]
        yeardelta = (dt.datetime.strptime(end,'%Y%m%d') - dt.datetime.strptime(start,'%Y%m%d')).days/365
        # 年化收益率
        ann_ret = pow(1 + MyFactor.cumret(df_ret['ret']),1/yeardelta) - 1
        return ann_ret

    @staticmethod
    def maxdrawdown(netval:Union[list,np.array,pd.Series])->float:
        netval = np.array(netval)
        drawdown = (np.maximum.accumulate(netval) - netval)/np.maximum.accumulate(netval)
        return max(drawdown)
    
    @staticmethod
    def stdstrdate(strdate:str)->str:
        '''将'%Y-%m-%d'格式的日期转为'%Y%m%d'格式的日期并填充0'''
        strdatelst = strdate.split('-')
        year = strdatelst[0].zfill(4)
        month = strdatelst[1].zfill(2)
        date = strdatelst[2].zfill(2)
        return year + month + date

    @staticmethod
    def newey_west_test(arr:Union[List,np.array,pd.Series],lag = None)->float:
        '''对序列做newey-test t检验,原假设为序列均值为0,备择假设为序列均值非0'''
        df_arr = pd.DataFrame(arr.values,columns = ['x'])
        df_arr['one'] = 1
        df_reg = MyFactor.newey_west_reg(df_arr,lag)
        if df_reg is None:
            return None
        return df_reg.loc['tval','one']
    
    @staticmethod
    def newey_west_reg(df:pd.DataFrame,lag:int = None)->pd.DataFrame:
        '''
        将df最左侧的向量向其右侧变量回归,计算回归的alpha,beta,newey-west t值

        参数:

        df:第1列为被解释变量,其余各列为解释变量

        lag:Newey-West滞后阶数,默认为int(4*(T/100)^(2/9))

        输出:

        df_newey_west:回归结果,共2行,列数为df列数-1
            第1行为回归系数,第2行为回归系数的Newey-West检验t值
        '''
        df = df.dropna()
        if len(df) == 0:
            return None
        lag = int(4*(len(df)/100)**(2/9)) if lag is None else lag
        y,X = df.columns[0],df.columns[1:]
        reg = smf.ols(f'{y}~' + '+'.join(X),data = df).fit(cov_type = 'HAC',cov_kwds = {'maxlags':lag})
        return pd.DataFrame([reg.params,reg.tvalues],columns = X,index = ['beta','tval'])

    @staticmethod
    def stdformat(df_factor:pd.DataFrame,trdcalendar:np.array)->pd.DataFrame:
        '''
        检查数据格式是否满足要求,去除日期或资产代码为空的数据行,将日期格式标准化为'%Y%m%d'

        将所有非交易日的日期转化为下一个最近的交易日,因子数据统一转化为float类型

        参数:

        df_factor:堆栈格式的因子表,size = [T*N,2 + factornum]

            第1列为日期,datatime.date或int或str,str格式为'%Y-%m-%d'或'%Y%m%d'(这时候不需要再改变格式)

            第2列为资产代码,str

            其余各列为因子值,float或bool

        trdcalendar:交易日历,trdcalendar的维数为1,values为str,格式为'%Y%m%d'

        输出:

        df_factor:格式化后的因子表,size = [T*N,2 + factornum]

            第1列为交易日,str,格式为'%Y%m%d',列名为trddate

            第2列为资产代码,str,列名为code

            其余各列为因子值,float
        '''
        ## 因子表日期标准化
        datename = df_factor.columns[0]
        codename = df_factor.columns[1]
        # 删除日期或资产代码为空值的行
        df_factor = df_factor.dropna(subset = [datename,codename])
        # 删除因子值全为空值的行
        df_factor = df_factor.dropna(subset = df_factor.columns[2:],how = 'all')
        df_factor = df_factor.reset_index(drop = True)
        # 日期数据格式转化成'%Y%m%d'
        if type(df_factor.loc[0,datename]) == dt.datetime.date:
            df_factor[datename] = df_factor.loc[:,datename].apply(lambda x: dt.datetime.strftime(x,'%Y$m%d'))
        elif (type(df_factor.loc[0,datename]) == str) and ('-' in df_factor.loc[0,datename]):
            df_factor[datename] = df_factor.loc[:,datename].apply(MyFactor.stdstrdate)
        else:
            df_factor[datename] = df_factor.loc[:,datename].apply(lambda x: str(x))
        ## 将原始日期转化为原始日期之后最近的交易日(当原始日期为交易日时直接用即可)
        df_factor[datename] = df_factor[datename].apply(lambda x: trdcalendar[(trdcalendar < x).sum()])
        ## 重新设置前两列列名(trddate,code)
        df_factor = df_factor.rename(columns = {datename:'trddate',codename:'code'})
        ## 根据trddate,code排序
        df_factor = df_factor.sort_values(['trddate','code']).reset_index(drop = True)
        # 查看trddate,code是否唯一
        if df_factor.duplicated(subset = ['trddate','code']).sum() != 0:
            print('Current factor got duplicated trddate and code, which could fail singlesort. You may need to drop duplicated rows.')
        ## 其余各列因子值统一转化为float格式
        df_factor.loc[:,df_factor.columns[2:]] = df_factor.loc[:,df_factor.columns[2:]].astype(float)
        ## 输出标准化后的因子表
        return df_factor

    @staticmethod
    def fillna(df_factor:pd.DataFrame,limit:int = 10)->pd.DataFrame:
        '''
        填充堆栈的非平衡面板缺失值(对于上市前的数据为空的情况不做填充,默认最多向后填充10条缺失数据)

        参数:

        df_factor:符合MyFactor.stdformat输出要求的因子表

        limit:前向填充缺失值的最大数量,默认为10条

        输出:

        df_factor:DataFrame,填充缺失值之后的df_factor,符合MyFactor.stdformat输出要求的因子表
        '''
        # 无缺失数据时的trddate*code
        newidx = set(product(set(df_factor['trddate']),set(df_factor['code'])))
        # 原始trddate*code
        df_factor = df_factor.set_index(['date','code'])
        # 增加的trddate*code
        df_factor = df_factor.reindex(newidx).sort_index()
        # 根据code分组填充,使用日期前面的的数据填充,最多填充10条
        df_factor = df_factor.groupby('code',group_keys = False).apply(lambda x: x.ffill(limit = limit))
        # 去除因子值全部空的行(一个因子也没填充成功的行)
        df_factor = df_factor.loc[~df_factor.isna().any(axis = 1)].reset_index()
        return df_factor

    @staticmethod
    def del_outlier(df_factor:pd.DataFrame,method:str,left:Union[int,float],right:Union[int,float],factorname:Union[str,list,None] = None)->pd.DataFrame:
        '''
        因子表去极值,共三种方法:mad法,sigma法,precentile法

        参数：

        df_factor:符合MyFactor.stdformat输出要求的因子表

        method:str,去极值方式,共3种:{'mad','sigma','percentile'}

            mad:根据数据中位数去极值

            sigma:根据数据方差去极值

            percentile:根据数据分位点去极值

        left,right:float,根据method确定的参数

            method == mad:mad方法两侧极值参数,阈值设置在几个绝对偏差中位数上(2,2)或(3,3)

            method == sigma:sigma方法两侧极值参数,阈值设置在几个标准差的地方,一般为3个标准差(3,3)

            method == percentile:不带百分号的percentile方法百分位数,两侧极值参数,百分位数,0到100的数(5,95)

        factorname:数据表列名,即因子名,默认为None(对因子表中的全部因子去极值)

        输出:

        df_factor:DataFrame,去极值后的数据,符合MyFactor.stdformat输出要求的因子表
        '''
        if factorname is None:
            factornamelst = list(df_factor.columns[2:])
        else:
            factornamelst = list(factorname)
        if method.lower() == 'mad':
            return df_factor.groupby('trddate', group_keys = False)[['trddate','code'] + factornamelst]\
                            .apply(lambda x: MyFactor.mad_del(x,left,right,factornamelst),include_groups = False)
        elif method.lower() == 'sigma':
            return df_factor.groupby('trddate', group_keys = False)[['trddate','code'] + factornamelst]\
                            .apply(lambda x: MyFactor.sigma_del(x,left,right,factornamelst),include_groups = False)
        elif method.lower() == 'percentile':
            return df_factor.groupby('trddate', group_keys = False)[['trddate','code'] + factornamelst]\
                            .apply(lambda x: MyFactor.percentile_del(x,left,right,factornamelst),include_groups = False)

    @staticmethod
    def mad_del(df_factor_t:pd.DataFrame,left:Union[int,float],right:Union[int,float],factornamelst:list)->pd.DataFrame:
        '''
        对某个日期的截面数据采用单期MAD法去极值

        输入变量:

        df_factor_t:符合MyFactor.stdformat输出要求的单期因子表(trddate相同)

        left,right:两侧极值参数,阈值设置在几个绝对偏差中位数上(2,2)或(3,3)

        factornamelst:需要去极值的列名
        
        输出变量:

        df_factor_t:去除极值后的DataFrame,符合MyFactor.stdformat输出要求的单期因子表(trddate相同)
        '''
        # 去除全部因子值当期中位数
        factor_median = np.nanmedian(df_factor_t[factornamelst],axis = 0)
        # 每个因子相对中位数偏离的绝对值
        bias_sr = abs(df_factor_t[factornamelst] - factor_median)
        # 偏离绝对值的中位数
        new_median = np.nanmedian(bias_sr,axis = 0)
        # 左右阈值
        dt_down = factor_median - left * new_median
        dt_up = factor_median + right * new_median
        # 缩尾
        df_factor_t[factornamelst] = df_factor_t[factornamelst].clip(dt_down,dt_up,axis = 1)
        # 输出
        return df_factor_t

    @staticmethod
    def sigma_del(df_factor_t:pd.DataFrame,left:Union[int,float],right:Union[int,float],factornamelst:list)->pd.DataFrame:
        '''
        对某个日期的截面数据采用单期sigma法去极值

        参数:

        df_factor_t:符合MyFactor.stdformat输出要求的单期因子表(trddate相同)

        left,right:两侧极值参数,阈值设置在几个标准差的地方,一般为3个标准差(3,3)

        factornamelst:需要去极值的列名

        输出:

        df_factor_t:去除极值后的DataFrame,符合MyFactor.stdformat输出要求的单期因子表(trddate相同)
        '''
        # 均值和标准差
        mean = np.nanmean(df_factor_t[factornamelst],axis = 0)
        std = np.nanstd(df_factor_t[factornamelst],axis = 0)
        # 左右阈值
        dt_down = mean - left * std
        dt_up = mean + right * std
        # 缩尾
        df_factor_t[factornamelst] = df_factor_t[factornamelst].clip(dt_down,dt_up,axis = 1)
        # 输出
        return df_factor_t

    @staticmethod
    def percentile_del(df_factor_t:pd.DataFrame,left:Union[int,float],right:Union[int,float],factornamelst:list)->pd.DataFrame:
        '''
        对某个日期的截面数据采用单期percentile法去极值

        参数:

        df_factor_t:符合MyFactor.stdformat输出要求的单期因子表(trddate相同)

        factornamelst:需要去极值的列名

        left,right:两侧极值参数,百分位数,0到100的数(5,95)

        输出:

        df_factor_t:去除极值后的DataFrame,符合MyFactor.stdformat输出要求的单期因子表(trddate相同)
        '''
        # 左右阈值
        dt_up = np.nanpercentile(df_factor_t[factornamelst],right,axis = 0)
        dt_down = np.nanpercentile(df_factor_t[factornamelst],left,axis = 0)
        # 缩尾
        df_factor_t[factornamelst] = df_factor_t[factornamelst].clip(dt_down,dt_up,axis = 1)
        # 输出
        return df_factor_t

    @staticmethod
    def standardize(df_factor:pd.DataFrame,factorname:Optional[Union[str,list]] = None,stdtype:str = 'standardize')->pd.DataFrame:
        '''
        股票或因子值的堆栈数据标准化或归一化

        参数:

        df_factor:符合MyFactor.stdformat输出要求的因子表

        factorname:数据表列名,即因子名,默认为None(对因子表中的全部因子标准化或归一化)

        stdtype:stdtype == 'standardize'时做标准化(z-score),stdtype == 'normalize'时做归一化

        输出:

        df_factor:标准化或归一化后的股票或因子值,符合MyFactor.stdformat输出要求的因子表
        '''
        if factorname is None:
            factornamelst = df_factor.columns[2:]
        else:
            factornamelst = list(factorname)
        df_factor[factornamelst] = df_factor.groupby('trddate',group_keys = False)[['trddate','code'] + factornamelst]\
                                            .apply(lambda x: MyFactor.standardize_t(x,factornamelst,stdtype),include_groups =False)
        return df_factor

    @staticmethod
    def standardize_t(df_factor_t:pd.DataFrame,factornamelst:list,stdtype:str)->pd.DataFrame:
        '''
        对相同日期不同股票或因子值的面板数据标准化或归一化
        
        参数:

        df_factor_t:符合MyFactor.stdformat输出要求的单期因子表(trddate相同)

        factornamelst:需要标准化的因子名列表,列名

        stdtype:stdtype == 'standardize'时做标准化(z-score),stdtype == 'normalize'时做归一化

        输出:

        df_factor_t:某日期标准化后的股票或因子值,shape = [N,]
        '''
        if stdtype == 'standardize':
            meanval = np.nanmean(df_factor_t[factornamelst])
            stdval = np.nanstd(df_factor_t[factornamelst])
            df_factor_t[factornamelst] = (df_factor_t[factornamelst] - meanval)/stdval
        else:
            maxval = np.nanmax(df_factor_t[factornamelst])
            minval = np.nanmin(df_factor_t[factornamelst])
            df_factor_t[factornamelst] = (df_factor_t[factornamelst] - minval)/(maxval - minval)
        return df_factor_t

    @staticmethod
    def downfreq(df_factor:pd.DataFrame,cycle:int,downtype:Union[str,List[str]],ignorena:bool = False)->pd.DataFrame:
        '''
        将堆栈较高频率的数据低频化(日度->周度,月度,季度)

        参数:

        df_factor:符合MyFactor.stdformat输出要求的因子表

        cycle:int,低频化时的取样周期,例如cycle = 5时重新取样的数据根据原始的5条数据(window = 5)计算出
            常见的cycle = 5(交易日日度->交易日周度),cycle = 20(交易日日度->交易日月度)

        downtype:低频化时各列数据的处理方法,values有{'start'(取窗口第一个值),'end'(取窗口最后一个值),'median'(取窗口数据中位数)
            'mean'(取窗口数据平均值),'common'(取窗口数据众数),'sum'(取窗口数据之和),'max'(取窗口数据最大值),'min'(取窗口数据最小值)}

        ignorena:低频化时是否忽略高频数据中的缺失值,ignorena = True时即使窗口中有空值依旧根据type_lst方法计算,ignore = False时
            除了'start'和'end'方法下可能不受影响外,其它结果均为空值

        输出:

        df_resample:低频化后的DataFrame,符合MyFactor.stdformat输出要求的因子表
        '''
        # 存储用于apply的参数
        factornamelst = list(df_factor.columns[2:])
        downfreqdict = dict(zip(factornamelst,list(downtype)))
        # 获取日期序列,用于选择低频化的时间节点(周期的最后一天)
        df_trddate = df_factor[['trddate']].drop_duplicates().sort_values(by = 'trddate').reset_index(drop = True)
        endidx = list(range(cycle - 1,len(df_trddate),cycle))
        df_trddate['endtrddate'] = None
        df_trddate.loc[endidx,'endtrddate'] = df_trddate.loc[endidx,'trddate']
        df_trddate = df_trddate.bfill().dropna()
        # 把时间节点的对应关系匹配到因子表上,多余的日期(边角料)删掉
        df_factor = df_factor.merge(df_trddate,on = 'trddate',how = 'inner')
        df_factor = df_factor.set_index(['trddate','endtrddate','code']).stack().reset_index()
        df_factor.columns = ['trddate','endtrddate','code','factorname','factorvalue']
        # 数据降频,去除降频后全部为空值的行
        tqdm.pandas(desc = 'Reduce Data Frequency')
        df_factor = df_factor.groupby(['endtrddate','code','factorname'],group_keys = False)[df_factor.columns].\
                            progress_apply(lambda x:MyFactor.downfreq_it(x,downfreqdict,ignorena,cycle,df_trddate),include_groups = False).dropna(how = 'all').reset_index(drop = True)
        # 转换成原格式,去掉columns的名字
        df_factor = pd.pivot(df_factor,index = ['trddate','code'],columns = 'factorname',values = 'factorvalue').reset_index()
        df_factor.columns.name = None
        return df_factor

    @staticmethod
    def downfreq_it(df_factor_it:pd.DataFrame,downfreqdict:Mapping[str,str],ignorena:bool,cycle:int,df_trddate:pd.DataFrame)->pd.DataFrame:
        '''
        (聚合函数,用于apply)在一个窗口内为计算降频后的数据
        
        参数:

        df_factor_it:单个code在一个窗口期内在一个因子上的取值,行数小于等于cycle(小于cycle是因为可能存在空值),共5列

            第1列为原始交易日,type = str

            第2列为匹配后的降频交易日,type = str

            第3列为资产代码,type = str

            第4列为因子名,type = str

            第5列为因子取值,type = float

        downfreqdict:因子名和降频方法(downtype)对应关系的字典

        ignorena:低频化时是否忽略高频数据中的缺失值,ignorena = True时即使窗口中有空值依旧根据type_lst方法计算,ignore = False时
            除了'start'和'end'方法下可能不受影响外,其它结果均为空值

        cycle:cycle:int,低频化时的取样周期,例如cycle = 5时重新取样的数据根据原始的5条数据(window = 5)计算出
            常见的cycle = 5(交易日日度->交易日周度),cycle = 20(交易日日度->交易日月度)

        df_trddate:完整的原始交易日和降频后交易日的对应关系表,共2列

            第1列为'trddate',按顺序排列的原始因子表交易日序列,type = str

            第2列为'endtrddate',原始因子表交易日匹配的降频后交易日,type = str
        
        输出:

        df_downfreq_it:单个code在一个窗口期内在一个因子上的取值,共1行4列

            第1列为匹配后的降频交易日,type = str

            第2列为资产代码,type = str

            第3列为因子名,type = str

            第4列为降频后因子取值,type = float
        '''
        # 分组指标
        endtrddate = df_factor_it['endtrddate'].values[0]
        code = df_factor_it['code'].values[0]
        factorname = df_factor_it['factorname'].values[0]
        # 根据factorname匹配低频化方式
        downtype = downfreqdict[factorname]
        # 股票当期数据条数小于cycle时当期数据缺失,在缺失处为其补充trddate空值,并为factorvalue填充空值
        if ignorena == False:
            if len(df_factor_it) < cycle:# 存在缺失日期时
                if downtype in ['start','end']:# 'start'和'end'方法下可能不受空值影响(因为头和尾可能非空),填充空值后再降频
                    addtrddates = list(set(df_trddate.loc[df_trddate['endtrddate'] == endtrddate,'trddate']) - set(df_factor_it['trddate']))
                    addrows = pd.DataFrame(product(addtrddates,[endtrddate],[code],[factorname],[np.nan]),columns = df_factor_it.columns)
                    df_factor_it = pd.concat([df_factor_it,addrows]).sort_values('trddate')
                else:# 其它情况下只要存在空值,输出的结果必定为空,可以跳过计算直接输出
                    return pd.DataFrame(columns = ['trddate','code','factorname','factorvalue'])
            else:# 不存在缺失日期时直接计算即可
                pass
        else:# 忽略缺失值时,不需要考虑是否填充,直接计算即可
            pass
        # 数据降频
        match downtype:
            case 'start':
                df_downfreq_it = pd.DataFrame([endtrddate,code,factorname,df_factor_it['factorvalue'].values[0]]).T
            case 'end':
                df_downfreq_it = pd.DataFrame([endtrddate,code,factorname,df_factor_it['factorvalue'].values[-1]]).T
            case 'median':
                df_downfreq_it = pd.DataFrame([endtrddate,code,factorname,np.median(df_factor_it['factorvalue'].values)]).T
            case 'mean':
                df_downfreq_it = pd.DataFrame([endtrddate,code,factorname,np.mean(df_factor_it['factorvalue'].values)]).T
            case 'common':
                df_downfreq_it = pd.DataFrame([endtrddate,code,factorname,mode(df_factor_it['factorvalue'].values)[0]]).T
            case 'sum':
                df_downfreq_it = pd.DataFrame([endtrddate,code,factorname,sum(df_factor_it['factorvalue'].values)]).T
            case 'max':
                df_downfreq_it = pd.DataFrame([endtrddate,code,factorname,max(df_factor_it['factorvalue'].values)]).T
            case 'min':
                df_downfreq_it = pd.DataFrame([endtrddate,code,factorname,min(df_factor_it['factorvalue'].values)]).T
            case _:
                print(f'{factorname}的downtype参数不符合要求,需要修改')
        df_downfreq_it.columns = ['trddate','code','factorname','factorvalue']
        # 输出
        return df_downfreq_it

    @staticmethod
    def matchret_t(df_window_t:pd.DataFrame,df_close:pd.DataFrame)->pd.DataFrame:
        '''
        (聚合函数,用于apply)根据当前再平衡日期,下一次再平衡日期匹配窗口期内股票的累计收益率
        
        参数:
            df_window_t:t时期的日期和下一次再平衡日期,共1行2列
                第1列为trddate:str,格式为'%Y%m%d'
                第2列为nxtrebalance:str,下一次再平衡日期,格式为'%Y%m%d'

            df_close:对应资产的收盘价数据,至少包含3列
                第1列为trddate:str,格式为'%Y%m%d'
                第2列为code:str,资产代码
                其中包含列'adjclose':资产复权收盘价

        输出:
            currret:df_close中包含的资产在trddate和nxtrebalance之间的收益率
        '''
        start,end = df_window_t['trddate'].values[0],df_window_t['nxtrebalance'].values[0]
        startclose = df_close.loc[df_close['trddate'] == start,['code','adjclose']]
        endclose = df_close.loc[df_close['trddate'] == end,['code','adjclose']]
        currret = startclose.rename(columns = {'adjclose':'startclose'}).\
                    merge(endclose.rename(columns = {'adjclose':'endclose'}),on = 'code',how = 'outer')
        currret['ret'] = currret['endclose']/currret['startclose'] - 1
        currret.loc[:,['trddate','nxtrebalance']] = start,end
        currret = currret[['trddate','nxtrebalance','code','ret']]
        return currret

    def getfactorname(self,tablename:Optional[str] = None)->Optional[list]:
        '''获取因子名,tablename = None时直接输出因子表名-因子名字典'''
        if tablename == None:
            return self.factortabledict
        elif tablename in self.factortabledict.keys():
            return self.factortabledict[tablename]
        else:
            print('目标因子表不存在')
            return None

    def gettrdcalendar(self,exchange:str)->np.array:
        '''
        获取交易日历

        参数:

        exchange:交易所简称,目前数据库中包含'sz','sh'两个交易所的交易日历

        输出:

        trdcalendar:交易日历,trdcalendar的维数为1,values为str,格式为'%Y%m%d'
        '''
        # 读取深交所交易日历
        conn = pymysql.connect(host = self.host,user = self.user,password = self.password,database = 'base',port = 3306,charset = 'utf8mb4')
        cursor = conn.cursor()
        cursor.execute(f'SELECT * FROM `{exchange}calendar`')
        trdcalendar = pd.DataFrame(cursor.fetchall(),dtype = str).values
        trdcalendar = trdcalendar.reshape(len(trdcalendar))
        cursor.close()
        conn.close()
        return trdcalendar

    def getfactortables(self)->Dict[str,List[str]]:
        '''获取全量因子表及因子名称'''
        conn = pymysql.connect(host = self.host,user = self.user,password = self.password,port = 3306,charset = 'utf8mb4')
        cursor = conn.cursor()
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'factor'")
        factortablelst = np.array(cursor.fetchall())
        factortablelst = factortablelst.reshape(len(factortablelst)).tolist()
        factortabledict = {}
        for tablename in factortablelst:
            cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_schema = 'factor' and table_name = '{tablename}'")
            factornamelst = np.array(cursor.fetchall())
            factornamelst = factornamelst.reshape(len(factornamelst)).tolist()
            factornamelst.remove('trddate')
            factornamelst.remove('code')
            factortabledict[tablename] = factornamelst
        cursor.close()
        conn.close()
        return factortabledict

    def getassettrd(self,trdlst:List[str],start:str = None,end:str = None)->pd.DataFrame:
        '''
        获取资产交易数据

        参数:

        trdlst:values为str,需要提取的列名列表,全部列名见base.stktrd

        start,end:读取数据的起始和结束日期,格式为'%Y%m%d'

        输出:

        df_trd:符合MyFactor.stdformat输出要求的资产交易表
        '''
        start = '19900101' if start is None else start
        end = dt.datetime.strftime(dt.datetime.today().date(),format = '%Y%m%d') if end is None else end
        trdstr = '`' + '`,`'.join(trdlst) + '`'
        # 读取深交所交易日历
        print(f'loading {self.assettype}trd data...')
        timestart = time()
        conn = pymysql.connect(host = self.host,user = self.user,password = self.password,database = 'base',port = 3306,charset = 'utf8mb4')
        cursor = conn.cursor()
        cursor.execute(f'SELECT `trddate`,`code`,{trdstr} FROM {self.assettype}trd WHERE `trddate` >= {start} and `trddate` <= {end} ORDER BY `trddate`,`code`')
        df_trd = pd.DataFrame(cursor.fetchall(),columns = ['trddate','code'] + trdlst)
        cursor.close()
        conn.close()
        df_trd[trdlst] = df_trd[trdlst].astype(float)
        print(f'{self.assettype}trd data loaded, {time() - timestart:.2f}s passed')
        return df_trd

    def savefactor(self,df_factor:pd.DataFrame,tablename:str,if_exist:str = 'append'):
        '''
        存储df_factor中的因子到数据库dbname的tablename表中
        
        参数：

        df_factor:符合MyFactor.stdformat输出要求的因子表

        tablename:要存储的数据表名

        if_exist:默认为'append'

            'append':当tablename表中存在同名因子,不改变原因子数据,仅增量更新,在原因子表的基础上加入新增日期的数据

            'replace':当tablename表中存在同名因子,删除原因子,存储df_factor中的因子
        '''
        print(f'Saving FactorTable {tablename}...')
        timestart = time()
        factornamelst = list(df_factor.columns[2:])
        # 分别为交易日、资产代码和因子列设置不同的存储数据类型
        dtypedict = {'trddate':VARCHAR(8),'code':VARCHAR(40)}
        dtypedict.update(dict(zip(factornamelst,[DECIMAL(20,4)]*len(factornamelst))))
        # 建立连接引擎
        engine = create_engine(f'mysql+mysqlconnector://{self.user}:{self.password}@{self.host}/factor')
        if tablename not in self.factortabledict.keys():
            # 数据库中无目标因子表
            df_factor.to_sql(tablename,con = engine,if_exists = 'replace',index = False)
            # 更新因子表字典
            self.factortabledict[tablename] = factornamelst
        elif len(self.getinoutside(tablename,factornamelst)[1]) == 0:
            # 数据库中有目标因子表但无新增列,此时无需更新因子表字典
            df_factor.to_sql(tablename,con = engine,if_exists = if_exist,index = False)
        elif if_exist == 'replace':
            # 数据库中有目标因子表且有新增列,但要求替换掉原始因子值(replace)
            df_factor.to_sql(tablename,con = engine,if_exists = 'replace',index = False)
            # 更新因子表字典
            self.factortabledict[tablename] = factornamelst
        else:# 数据库中有目标因子表且有新增列,且要求在原始因子值基础上追加(append)
            # 这里因为to_sql不能新增列,所以当savefactor使用append时手动追加,
            # 将追加后的完整数据replace到原始表上,这样的坏处是更新一个很大的表时会非常慢
            insidelst,outsidelst = self.getinoutside(tablename,factornamelst)
            df_outsidefactor = df_factor[['trddate','code'] + outsidelst]
            df_dbfactor = self.loadfactor(tablename)
            # 先left拼接原始因子值,将拼接后的因子值(extended factor)replace到数据库中,保证原始因子值不变
            df_extendfactor = df_dbfactor.merge(df_outsidefactor,on = ['trddate','code'],how = 'left')
            df_extendfactor.to_sql(tablename,con = engine,if_exists = 'replace',index = False)
            # 手动取出新增的trddate和code
            adddatecode = pd.concat([df_factor[['trddate','code']],df_dbfactor[['trddate','code']]]).drop_duplicates(keep = False)
            if len(adddatecode) != 0:
                # 单独取出新增的行
                df_addfactor = df_factor.merge(adddatecode,on = ['trddate','code'],how = 'inner')
                # 再增量append新的因子值(新的日期和股票代码)
                df_addfactor.to_sql(tablename,con = engine,if_exists = 'append',index = False)
            # 更新因子表字典
            self.factortabledict[tablename] = insidelst + outsidelst
        # 手动调整因子表各列的格式
        self.stdmysqltable(tablename)
        # 输出格式标准化后存储的因子表
        print(f'FactorTable {tablename} Saved, {time() - timestart:.2f}s passed')

    def stdmysqltable(self,tablename:str):
        '''to_sql直接存储的table格式有瑕疵,手动调整因子表各列格式,并设置INDEX'''
        conn = pymysql.connect(host = self.host,user = self.user,password = self.password,database = 'factor',port = 3306,charset = 'utf8mb4')
        cursor = conn.cursor()
        # 修改各列格式
        cursor.execute(f"ALTER TABLE `{tablename}` CHANGE `trddate` `trddate` varchar(8)")
        cursor.execute(f"ALTER TABLE `{tablename}` CHANGE `code` `code` varchar(40)")
        for factorname in self.factortabledict[tablename]:# 这时候因子表字典已经完成更新,所以可以直接用字典中存储的因子
            cursor.execute(f"ALTER TABLE `{tablename}` CHANGE `{factorname}` `{factorname}` DECIMAL(20,4)")
        # 把trddate和code设置为index
        cursor.execute(f"ALTER TABLE `{tablename}` ADD INDEX (`trddate`)")
        cursor.execute(f"ALTER TABLE `{tablename}` ADD INDEX (`code`)")
        # 统一提交更改
        conn.commit()
        # 关闭连接
        cursor.close()
        conn.close()

    def loadfactor(self,tablename:str,factorname:Optional[Union[str,list]] = None,start:Optional[Union[str,int]] = None,end:Optional[Union[str,int]] = None)->pd.DataFrame:
        '''
        下载格式化后的因子表

        参数:

        tablename:数据表名

        factorname:数据表列名,即因子名,默认为None(提取因子表中的全部因子)

        start:起始日期,格式为'%Y%m%d',默认为None(不设置起始日期)

        end:终止日期,格式为'%Y%m%d',默认为None(不设置终止日期)

        输出:

        df_factor:符合MyFactor.stdformat输出要求的因子表
        '''
        if tablename not in self.factortabledict.keys():
            print('目标因子表不存在')
            return None
        else:
            if factorname is None:
                return self.getfactortable(tablename,self.factortabledict[tablename],start,end)
            else:
                factornamelst = list(factorname)
        # 判断需要的因子是否在因子表中
        insidelst,outsidelst = self.getinoutside(tablename,factornamelst)
        if len(insidelst) == 0:
            print('目标因子表中无需要的因子')
            return None
        elif len(outsidelst) != 0:
            lostfactor = ''
            for factorname in outsidelst:
                lostfactor += f'{factorname},'
            print(f'因子表{tablename}中{lostfactor[:-1]}因子缺失')
        return self.getfactortable(tablename,insidelst,start,end)

    def getinoutside(self,tablename:str,factornamelst:List[str])->List[List[str]]:
        '''获取输入的因子名列表(factornamelst)与因子表中原始因子名列表的交集,factornamelst有而原始因子名列表没有的因子'''
        dbfactorlst = self.factortabledict[tablename]
        insidelst = list(set(dbfactorlst) & set(factornamelst))
        outsidelst = list(set(factornamelst) - set(dbfactorlst))
        return insidelst,outsidelst

    def getfactortable(self,tablename:str,factornamelst:list,start:Optional[Union[str,int]] = None,end:Optional[Union[str,int]] = None)->pd.DataFrame:
        '''
        读取因子表因子数据并输出为DataFarme
        
        参数:

        tablename:因子表名

        factornamelst:因子名列表

        start:起始日期,格式为'%Y%m%d',默认为None(不设置起始日期)

        end:终止日期,格式为'%Y%m%d',默认为None(不设置终止日期)

        输出:

        df_factor:符合MyFactor.stdformat输出要求的因子表
        '''
        conn = pymysql.connect(host = self.host,user = self.user,password = self.password,database = 'factor',port = 3306,charset = 'utf8mb4')
        cursor = conn.cursor()
        factornames = ''
        for factorname in factornamelst:
            factornames += f'`{factorname}`,'
        factornames = factornames[:-1]
        if start == None:
            start = '19000101'
        if end == None:
            end = '99990101'
        start,end = str(start),str(end)
        cursor.execute(f'SELECT `trddate`,`code`,{factornames} FROM `{tablename}` WHERE `trddate` > {start} and `trddate` <= {end}')
        df_factor = pd.DataFrame(cursor.fetchall(),columns = ['trddate','code'] + factornamelst)
        df_factor[factornamelst] = df_factor[factornamelst].astype(float)
        cursor.close()
        conn.close()
        return df_factor

    def matchret(self,df:pd.DataFrame)->pd.DataFrame:
        '''
        为包含trddate,nxtrebalance,code的堆栈数据表拼接收盘价,并计算下一期收益率,新表相比原表增添一列ret,其值为下一期收益率
        
        参数:

            df:包含code,trddate,nxtrebalance这3列,其中
                code:str,资产代码
                trddate:str,格式为'%Y%m%d',当期交易日
                nxtrebalancestr,格式为'%Y%m%d',下一次再平衡日期
        
        输出:

            df:原列均不变,仅仅根据df的trddate,nxtrebalance,code计算了每个资产的下一期收益率,新增1列ret
        '''
        # 取出收盘价,去掉日期不是再平衡交易日的数据
        df_window = df[['trddate','nxtrebalance']].drop_duplicates().reset_index(drop = True)
        rebalancelst = list(set(df_window['trddate'])|set(df_window['nxtrebalance']))
        df_close = self.df_trd.loc[self.df_trd['trddate'].isin(rebalancelst),['trddate','code','adjclose']].dropna()
        # 找到再平衡交易日后,用再平衡交易日匹配收益率,此时df_factor每一行都是再平衡之前的因子值和再平衡之后时期的收益率
        tqdm.pandas(desc = 'match rebalance ret')
        df_window = df_window.groupby(['trddate','nxtrebalance'],group_keys = False)[['trddate','nxtrebalance']].\
                        apply(lambda x: MyFactor.matchret_t(x,df_close),include_groups = False).dropna().reset_index(drop = True)
        df_window.columns = ['trddate','nxtrebalance','code','ret']
        df = df.merge(df_window,on = ['trddate','nxtrebalance','code'],how = 'left')
        return df

    def fama_macbeth(self,df_factor:pd.DataFrame,lag:Optional[int] = None)->FamaMacBeth:
        '''
        根据输入的因子表将收益率向全部因子做Fama-Mecbeth回归
        
        参数:

            df_factor:符合MyFactor.stdformat输出要求的因子表

                第1列为trddate
            
                第2列为code
        
                第3~end列为fama-macbeth回归的解释变量
        
        输出:

            famamacbeth:linearmodels.FamaMecbeth对象
        '''
        if self.df_trd is None:
            raise ValueError('Base trade data not loaded, unable to execute fama-macbeth.')
        # 因子列表
        factorlst = list(df_factor.columns[2:])
        # 找出每行的下一次再平衡交易日(这里直接用股票对应因子下一期日期即可)
        df_factor.loc[:,'nxtrebalance'] = df_factor.groupby('code',group_keys = False)['trddate'].apply(lambda x: x.shift(-1))
        df_factor = df_factor.dropna(subset = 'nxtrebalance').reset_index(drop = True)
        # 根据trddate和nxtrebalence获取下一期收益率
        df_factor = self.matchret(df_factor)
        df_factor = df_factor.loc[:,['trddate','code','ret'] + factorlst].dropna().reset_index(drop = True)
        # 调整日期数据类型
        df_factor['trddate'] = pd.to_datetime(df_factor['trddate'])
        # 计算滞后项阶数
        lag = int(4*(len(df_factor['trddate'].drop_duplicates())/100)**(2/9)) if lag is None else lag
        # 设置双重index,注意索引顺序
        df_factor.set_index(['code','trddate'],inplace = True)
        # 写出回归方程
        formula = 'ret~1+' + '+'.join(factorlst)
        # Fama-MacBeth回归
        mod = FamaMacBeth.from_formula(formula,data = df_factor)
        famamacbeth = mod.fit(cov_type = 'kernel',debiased = False,bandwidth = lag)
        return famamacbeth

    def singlesort(self,df_factor:pd.DataFrame,g:int = 5,rebalance:Optional[Union[List[str],np.array]] = None,weight:Optional[pd.DataFrame] = None,ascending:Optional[List[bool]] = None,fee:float = 0.003):
        '''
        根据输入的因子表对资产做单分组,根据输入的rebalancedate和weight计算各期的分组收益率

        参数:

        df_factor:符合MyFactor.stdformat输出要求的因子表

        g:int,分组个数,默认分5组

        rebalance:资产组合再平衡时间

            None:将df_factor中每个trddate的下一个交易日作为再平衡日期

            list:直接设定再平衡日期,当其中出现非交易日时使用当日最近的下一个交易日再平衡,values为str类型的日期,格式为'%Y%m%d'

        weight:等权重和市值加权之外的的组内权重表

            None:计算等权重分组收益率和流通市值(基金时使用总规模)加权分组收益率

            DataFrame:格式同MyFactor.stdformat输出要求,'trddate'和'code'之外的各列是资产加权的权重,
                    此时计算等权重,流通市值加权和weight中的权重加权分组收益率

        ascending:原始因子是否是正向因子,用于改变原始因子正负号

            None:全部因子都是正向预期因子(因子值越大,收益率越高)

            list:各因子是否是正向预期因子,与df_factor的因子顺序一致
        
        fee:费率,默认双边千分之三

        输出:

        SingleSort:SingleSort对象,单分组的结果
        '''
        self.g = g
        self.factornamelst = list(df_factor.columns[2:])
        # 引入交易日和资产交易数据,将因子表和交易数据,再平衡日期,分组权重相互匹配
        df_factor = self.matchsinglesort(df_factor,rebalance,weight,ascending)
        # ICIR
        df_ic = df_factor.groupby('trddate',group_keys = False)[['trddate','code','ret'] + self.factornamelst].apply(self.getic_t,include_groups = False)
        df_avgic = df_ic.groupby('factorname',group_keys = False)[[f'{self.assettype}num','ic','rankic']].apply(lambda x: x.mean()).reset_index()
        df_avgic.columns = ['factorname',f'avg{self.assettype}num','avgic','avgrankic']
        df_icir = df_ic.groupby('factorname',group_keys = False)[['ic','rankic']].apply(lambda x: x.mean()/x.std()).reset_index()
        df_icir.columns = ['factorname','ir','rankir']
        df_icir = df_avgic.merge(df_icir,on = 'factorname',how = 'left')
        # 因子相关系数矩阵
        factorcorr = df_factor.groupby('trddate',group_keys = False)[['trddate'] + self.factornamelst].apply(self.getcorr_t,include_groups = False)
        factorcorr = factorcorr.groupby('factorname',group_keys = False)[self.factornamelst].mean().reset_index()
        # 分组并计算收益率
        singlesort_id_dict,singlesort_ret_dict,singlesort_netval_dict = dict(),dict(),dict()
        singlesort_fee_dict,singlesort_trdret_dict,singlesort_trdnetval_dict = dict(),dict(),dict()
        lastweightlst,dweightlst = [f'last{weightname}' for weightname in self.weightlst],[f'd{weightname}' for weightname in self.weightlst]
        feeweightlst,trdweightlst = [f'fee{weightname}' for weightname in self.weightlst],[f'trd{weightname}' for weightname in self.weightlst]
        renamedict = dict(zip(['trddate','nxttrddate'] + self.weightlst,['lasttrddate','trddate'] + lastweightlst))
        df_ratios = pd.DataFrame(columns = ['factorname','weight','id','annret','anntrdret','retmaxdrawdown','trdretmaxdrawdown','rettval','trdrettval'])
        idlst = [str(p) for p in range(1,self.g + 1)] + ['longshort']
        for factorname in self.factornamelst:
            # 取出仅包含单个因子的因子表
            df_factor_ = df_factor[['trddate','code','ret'] + self.weightlst + [factorname]].dropna().reset_index(drop = True)
            # 计算分组结果
            df_id = df_factor_.groupby('trddate',group_keys = False)\
                        [['trddate','code','ret'] + self.weightlst + [factorname]].apply(self.singlesort_id_t,include_groups = False)
            # 根据分组结果计算多空组合股票权重
            df_factor_longshort = df_id.loc[df_id['id'].isin([str(self.g),'1']),:]
            # 空头部分权重乘-1
            df_factor_longshort.loc[df_factor_longshort['id'] == '1',self.weightlst] = -df_factor_longshort.loc[df_factor_longshort['id'] == '1',self.weightlst]
            # 合并相同资产的多空权重
            df_factor_longshort = df_factor_longshort.groupby(['trddate','code','ret'],group_keys = False)[self.weightlst].sum().reset_index()
            df_factor_longshort['id'] = 'longshort'
            # 把多空权重合并到原组合权重上
            df_id = pd.concat([df_id,df_factor_longshort]).sort_values(['trddate','id','code'])
            # 计算分组收益率
            df_ret = df_id.groupby('trddate',group_keys = False)[['trddate','code','id','ret'] + self.weightlst].\
                                    apply(self.singlesort_ret_t,include_groups = False).sort_values(['trddate','id'])
            # 分组累计净值
            df_netval = df_ret.groupby('id',group_keys = False)[['trddate','id'] + self.weightlst].\
                                    apply(self.getnetval_id,include_groups = False).sort_values(['trddate','id'])
            # 根据当期权重和上一期权重计算前后两期各资产权重变动,这里需要精确匹配每只股票前一期的权重
            df_datematch = pd.DataFrame(df_id['trddate'].drop_duplicates().sort_values(),columns = ['trddate'])
            df_datematch['lasttrddate'] = df_datematch['trddate'].shift(1)
            df_datematch['nxttrddate'] = df_datematch['trddate'].shift(-1)
            # 为原始权重匹配前一期的权重
            df_idfee = df_id.merge(df_datematch,on = 'trddate',how = 'left')
            df_idmatch = df_idfee[['nxttrddate','trddate','code','id'] + self.weightlst].rename(columns = renamedict)
            df_idfee = df_idfee.merge(df_idmatch,on = ['trddate','lasttrddate','code','id'],how = 'outer').dropna(subset = 'trddate')
            df_idfee = df_idfee.drop(['lasttrddate','nxttrddate'],axis = 1).fillna(0)
            # 计算前后两期权重变动的绝对值并求和,用于计算费用
            df_idfee.loc[:,dweightlst] = np.abs(df_idfee.loc[:,self.weightlst].values - df_idfee.loc[:,lastweightlst].values)
            df_fee = df_idfee.groupby(['trddate','id'],group_keys = False)[dweightlst].sum().reset_index()
            df_fee.loc[:,dweightlst] = df_idfee.loc[:,dweightlst]*fee
            df_fee = df_fee.rename(columns = dict(zip(dweightlst,feeweightlst))).sort_values(['trddate','id'])
            # 把费率表和分组收益率表合并,计算出考虑费率的组合收益率
            df_trdret = df_ret.merge(df_fee,on = ['trddate','id'],how = 'left')
            df_trdret.loc[:,trdweightlst] = (1 + df_trdret.loc[:,self.weightlst].values)*(1 - df_trdret.loc[:,feeweightlst].values) - 1
            df_trdret = df_trdret.loc[:,['trddate','id'] + trdweightlst].sort_values(['trddate','id'])
            # 考虑交易费用的分组累计净值
            df_trdnetval = df_trdret.groupby('id',group_keys = False)[['trddate','id'] + trdweightlst].\
                                    apply(self.getnetval_id,include_groups = False).sort_values(['trddate','id'])
            # 存储分组结果和分组收益率
            singlesort_id_dict[factorname] = df_id[['trddate','code','id'] + self.weightlst]
            singlesort_ret_dict[factorname] = df_ret
            singlesort_netval_dict[factorname] = df_netval
            singlesort_fee_dict[factorname] = df_fee
            singlesort_trdret_dict[factorname] = df_trdret
            singlesort_trdnetval_dict[factorname] = df_trdnetval
            for weight in self.weightlst:
                for id in idlst:
                    # 读取收益率和净值数据
                    df_ret_id = df_ret.loc[df_ret['id'] == id,['trddate'] + [weight]]
                    df_trdret_id = df_trdret.loc[df_trdret['id'] == id,['trddate'] + [f'trd{weight}']]
                    df_netval_id = df_netval.loc[df_netval['id'] == id,['trddate'] + [weight]]
                    df_trdnetval_id = df_trdnetval.loc[df_trdnetval['id'] == id,['trddate'] + [f'trd{weight}']]
                    # 年化收益率
                    ann_ret = MyFactor.ret2annual(df_ret_id[['trddate',weight]])
                    ann_trdret = MyFactor.ret2annual(df_trdret_id[['trddate',f'trd{weight}']])
                    # 最大回撤
                    maxdrawdown_ret = MyFactor.maxdrawdown(df_netval_id[weight])
                    maxdrawdown_trdret = MyFactor.maxdrawdown(df_trdnetval_id[f'trd{weight}'])
                    # t值
                    tval_ret = MyFactor.newey_west_test(df_ret_id[weight])
                    tval_trdret = MyFactor.newey_west_test(df_trdret_id[f'trd{weight}'])
                    # 存储结果
                    df_ratios.loc[len(df_ratios)] = [factorname,weight,id,ann_ret,ann_trdret,maxdrawdown_ret,maxdrawdown_trdret,tval_ret,tval_trdret]
        return SingleSort(df_ic,df_icir,df_ratios,factorcorr,singlesort_id_dict,singlesort_ret_dict,singlesort_netval_dict,singlesort_fee_dict,singlesort_trdret_dict,singlesort_trdnetval_dict)

    def getnetval_id(self,df_ret_id)->pd.DataFrame:
        '''
        (聚合函数,用于apply)计算一个分组的资产组合净值

        参数:
            df_ret_id:一个分组的收益率表,各行id相等
                第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
                第2列为id:str,1~g的分组结果及longshort
                第3~len(self.weightlst)+3列为self.weightlst:float,自定义加权的分组收益率
        
        输出:
            df_netval_id:一个分组的资产组合净值表,各行id相等
                第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
                第2列为id:str,1~g的分组结果及longshort
                第3~len(self.weightlst)+3列为self.weightlst:float,自定义加权的分组净值
        '''
        weightlst = list(df_ret_id.columns[2:])
        df_netval_id = df_ret_id.loc[:,['trddate','id']]
        df_netval_id[weightlst] = np.cumprod(1 + df_ret_id[weightlst])
        df_netval_id[weightlst] = df_netval_id[weightlst].shift(1).fillna(1)
        return df_netval_id

    def getcorr_t(self,df_factor_t:pd.DataFrame)->pd.DataFrame:
        '''
        (聚合函数,用于apply)计算因子截面相关系数

        参数:
            df_factor_t:包含多个因子值的单期因子表,各行trddate相等
                第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
                第2~2+len(self.factornamelst)列为正向的因子值:float
        
        输出:
            df_corr_t:单期的因子相关系数矩阵,各行trddate相等,共len(self.factornamelst)行
                第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
                第2列为self.factornamelst:str,因子名
                第3~3+len(self.factornamelst)列为因子截面相关系数:float
        '''
        trddate = df_factor_t['trddate'].values[0]
        df_corr_t = df_factor_t[self.factornamelst].corr().reset_index()
        df_corr_t.columns = ['factorname'] + self.factornamelst
        df_corr_t['trddate'] = trddate
        df_corr_t = df_corr_t[['trddate','factorname'] + self.factornamelst]
        return df_corr_t

    def getic_t(self,df_factor_t:pd.DataFrame)->pd.DataFrame:
        '''
        (聚合函数,用于apply)获取因子单期的IC

        参数:
            df_factor_t:包含多个因子的单期因子表,结构与matchsinglesort输出结果的结构一致,但各行trddate相等
                第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
                第2列为code:str,资产代码
                第3列为ret:float,资产从当前再平衡日期开始,到下一个再平衡日期截止的累计收益率
                第4~4+len(self.factornamelst)列为正向的因子值:float
        
        输出:
            df_ic_t:多个因子的单期ic表,共5列
                第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
                第2列为factorname:str,因子名
                第3列为当期完整的股票和因子数据条数:int
                第4列为ic:float,当期的因子ic
                第5列为rankic:float,当期因子rankic
        '''
        df_ic_t = pd.DataFrame(np.full([len(self.factornamelst),5],None),columns = ['trddate','factorname',f'{self.assettype}num','ic','rankic'])
        df_ic_t.loc[:,'trddate'] = df_factor_t['trddate'].values[0]
        df_ic_t.loc[:,'factorname'] = self.factornamelst
        df_ic_t.loc[:,f'{self.assettype}num'] = 0
        for i in range(len(self.factornamelst)):
            factorname = self.factornamelst[i]
            curr_factor_ret = df_factor_t.loc[:,['ret',factorname]].dropna()
            if len(curr_factor_ret) != 0:
                df_ic_t.iloc[i,2] = len(curr_factor_ret)
                df_ic_t.iloc[i,3] = curr_factor_ret.corr().iloc[1,0]
                df_ic_t.iloc[i,4] = curr_factor_ret.corr('spearman').iloc[1,0]
        return df_ic_t

    def singlesort_id_t(self,df_factor_t:pd.DataFrame)->pd.DataFrame:
        '''
        (聚合函数,用于apply)在截面上根据单个因子对股票分组并输出分组结果和归一化的组内权重
        
        参数:
            df_factor_t:包含一个因子的单期因子表,结构与matchsinglesort输出结果的结构一致,但仅包含1个因子且各行trddate相等
                第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
                第2列为code:str,资产代码
                第3列为ret:float,资产从当前再平衡日期开始,到下一个再平衡日期截止的累计收益率
                第4列为single:int,全部为1,组内等权重加权的权重
                第5列为size:float,资产市值,组内市值加权的权重
                [weight不为None时]第6~np.size(weight,1)+6列为weight.columns[2:]:float,组内自定义方式加权的权重
                第np.size(weight,1)+7列为正向的因子值

        输出:
            df_id_t:资产分组结果和归一化后的权重表
                第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
                第2列为code:str,资产代码
                第3列为id:int,1~g的分组结果
                第4列为ret:float,资产从当前再平衡日期开始,到下一个再平衡日期截止的累计收益率
                第5列为single:int,相同id全部相等,归一化的组内等权重加权的权重
                第6列为size:float,归一化的组内市值加权的权重
                [weight不为None时]第7~np.size(weight,1)+7列为weight.columns[2:]:float,归一化的组内自定义方式加权的权重
        '''
        factorname = df_factor_t.columns[-1]
        # 分为g组，共g+1个分割点(包括最大值和最小值)
        percentile = np.percentile(df_factor_t[factorname],[100/self.g*i for i in range(self.g+1)])
        if len(set(percentile)) < self.g + 1:
            # 无法分组时是因为同一时期资产数量过少或大量因子值相同,此时将收益率全部置为0并随机分组
            print(f'Single sort fail to split {factorname} {df_factor_t['trddate'].values[0]} data into {self.g} parts, please check factor value or decrease group num.')
            df_factor_t.loc[:,'ret'] = 0
            df_factor_t.loc[:,'id'] = np.random.randint(1,self.g + 1,len(df_factor_t)).astype('str')
        else:# 分组
            df_factor_t.loc[:,'id'] = pd.cut(df_factor_t[factorname],include_lowest = True,bins = percentile,labels = np.arange(1,self.g+1)).astype('str')
        # 权重归一化
        df_id_t = df_factor_t.groupby('id',group_keys = False)[['trddate','code','id','ret'] + self.weightlst].apply(self.weightnormalize,include_groups = False)
        return df_id_t

    def singlesort_ret_t(self,df_id_t:pd.DataFrame)->pd.DataFrame:
        '''
        (聚合函数,用于apply)在截面上根据单个因子对股票的分组结果计算分组收益率

        参数:
            df_id_t:singlesort_id_t的输出,资产分组结果和归一化后的权重表
                第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
                第2列为code:str,资产代码
                第3列为id:int,1~g的分组结果
                第4列为ret:float,资产从当前再平衡日期开始,到下一个再平衡日期截止的累计收益率
                第5列为single:int,相同id全部相等,归一化的组内等权重加权的权重
                第6列为size:float,归一化的组内市值加权的权重
                [weight不为None时]第7~np.size(weight,1)+7列为weight.columns[2:]:float,归一化的组内自定义方式加权的权重

        输出:
            df_ret_t:分组收益率
                第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
                第2列为id:str,1~g的分组结果及longshort
                第3列为single:float,等权重加权的分组收益率
                第4列为size:float,市值加权的分组收益率
                [weight不为None时]第5~np.size(weight,1)+5列为weight.columns[2:]:float,自定义加权的分组收益率
        '''
        # 根据分组结果group,计算分组收益率
        df_ret_t = df_id_t.groupby('id',group_keys = False)[['trddate','id','ret'] + self.weightlst].\
                        apply(self.retgroup,include_groups = False).reset_index(drop = True)
        df_ret_t = df_ret_t[['trddate','id'] + self.weightlst]
        return df_ret_t

    def weightnormalize(self,factor_weight:pd.DataFrame)->pd.DataFrame:
        '''(聚合函数,用于apply)将同组内的weight之和收缩到1'''
        factor_weight.loc[:,self.weightlst] = factor_weight[self.weightlst]/factor_weight[self.weightlst].sum()
        return factor_weight

    def retgroup(self,factor_ret:pd.DataFrame)->pd.DataFrame:
        '''(聚合函数,用于apply)计算分组收益率'''
        trddate,id = factor_ret['trddate'].values[0],factor_ret['id'].values[0]
        factor_group = factor_ret[['ret']].T@factor_ret[self.weightlst]
        factor_group[['trddate','id']] = trddate,id
        return factor_group

    def matchsinglesort(self,df_factor:pd.DataFrame,rebalance:Optional[Union[List[str],np.array]] = None,weight:Optional[pd.DataFrame] = None,ascending:Optional[List[bool]] = None)->pd.DataFrame:
        '''
        为因子表匹配再平衡交易日,分组权重,收益率,为单分组和计算ICIR做准备

        参数:
        df_factor:符合MyFactor.stdformat输出要求的因子表
        rebalance:None or list,资产组合再平衡时间,默认为None
            None:将df_factor中每个trddate的下一个交易日作为再平衡日期
            list:直接设定再平衡日期,当其中出现非交易日时使用当日最近的下一个交易日再平衡,values为str类型的日期,格式为'%Y%m%d'
        weight:None or DataFrame,默认为None
            None:计算等权重分组收益率和流通市值(基金时使用总规模)加权分组收益率
            DataFrame:格式同MyFactor.stdformat输出要求,'trddate'和'code'之外的各列是资产加权的权重,trddate需要包含每次再平衡前一个交易日
                    此时计算等权重,流通市值加权和weight中的权重加权分组收益率
        ascending:None or list,默认为None
            None:全部因子都是正向预期因子(因子值越大,收益率越高)
            list:各因子是否是正向预期因子,与df_factor的因子顺序一致

        输出:
        df_factor:符合MyFactor.stdformat输出要求的因子表,非正向因子全部转为正向因子,共2(trddate,code) + 2(组内等权重,组内市值加权) + weight数量 + factor数量 + 1(ret)列
            第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
            第2列为code:str,资产代码
            第3列为ret:float,资产从当前再平衡日期开始,到下一个再平衡日期截止的累计收益率
            第4列为single:int,全部为1,组内等权重加权的权重
            第5列为size:float,资产市值,组内市值加权的权重
            [weight不为None时]第6~np.size(weight,1)+6列为weight.columns:float,组内自定义方式加权的权重
            第np.size(weight,1)+7~np.size(df_factor.loc[:,2:])+np.size(weight,1)+7列为df_factor.columns[2:]:float,转为正向的因子值
        '''
        if self.df_trd is None:
            raise ValueError('Base trade data not loaded, unable to execute singlesort.')
        # 计算全部加权方法下的权重
        df_weight = self.df_trd.loc[:,['trddate','code','size']]
        df_weight.loc[:,'single'] = 1.0
        df_weight = df_weight[['trddate','code','single','size']]
        self.weightlst = ['single','size']
        if weight is not None:
            df_weight = df_weight.merge(weight,on = ['trddate','code'],how = 'left')
            self.weightlst = self.weightlst + list(weight.columns[2:])
        # 根据再平衡日期表的情况为因子表匹配不同的再平衡交易日
        if rebalance is None:# 参数中无再平衡日期时,因子表每个交易日的下一个交易日再平衡
            # 再平衡前1个交易日计算权重,这里因为再平衡交易日正好是因子交易日的后一个交易日,所以直接用因子交易日匹配即可
            df_factor = df_factor.merge(df_weight,on = ['trddate','code'],how = 'left')
            # 每个交易日的下一个交易日,用于匹配再平衡日期
            trddatematch = pd.DataFrame(self.trdcalendar,dtype = str,columns = ['trddate'])
            trddatematch['nxttrddate'] = trddatematch['trddate'].shift(-1)
            trddatematch = trddatematch.dropna()
            # 把全部因子交易日都换成再平衡交易日
            df_factor = df_factor.merge(trddatematch,on = 'trddate',how = 'left').drop('trddate',axis = 1).rename(columns = {'nxttrddate':'trddate'})
            # 找到当期再平衡交易日后,再匹配下一个最近的再平衡交易日,用于计算两期之间的累计收益率
            rebalancematch = pd.DataFrame(df_factor['trddate'].drop_duplicates().sort_values())
            rebalancematch['nxtrebalance'] = rebalancematch['trddate'].shift(-1)
            df_factor = df_factor.merge(rebalancematch,on = 'trddate',how = 'left')
            # 去掉nxtrebalance为空的行,最后一天的因子值不用做回测
            df_factor = df_factor.dropna(subset = ['nxtrebalance'])
        else:
            # 参数中有再平衡日期时,先将再平衡日期换成后一个最近的交易日(含当天)
            rebalancematch = pd.DataFrame(rebalance,dtype = str,columns = ['trddate']).sort_values(by = 'trddate')
            rebalancematch['trddate'] = rebalancematch['trddate'].apply(lambda x: self.trdcalendar[(self.trdcalendar < x).sum()])
            # 获取再平衡交易日的前一个交易日,用于在再平衡前1天计算组内权重
            rebalancematch['weightdate'] = rebalancematch['trddate'].apply(lambda x: self.trdcalendar[(self.trdcalendar < x).sum() - 1])
            # 匹配每一个再平衡交易日的下一个最近的再平衡交易日,用于计算两期之间的累计收益率
            rebalancematch['nxtrebalance'] = rebalancematch['trddate'].shift(-1)
            rebalancematch = rebalancematch.dropna()
            # 用每个因子的交易日和下一个交易日,限定匹配再平衡交易日的范围.某一期股票因子数据缺失时,
            # 这样做而不是在股票上group再shift可以确保缺失之前的因子数据不会被错误地用于当前分组
            factordate = df_factor[['trddate']].drop_duplicates().sort_values(by = 'trddate')
            factordate['nxtfactor'] = factordate.shift(-1)
            df_factor = df_factor.merge(factordate,on = 'trddate',how = 'left')
            df_factor = df_factor.dropna(subset = 'nxtfactor')
            # 为每一行数据匹配再平衡交易日
            tqdm.pandas(desc = 'match rebalance date')
            df_factor = df_factor.groupby(['trddate','code'],group_keys = False)[df_factor.columns].\
                            progress_apply(lambda x: self.matchrebalence(x,rebalancematch),include_groups = False)
            # 匹配权重表,再平衡前1天计算权重
            df_factor = df_factor.merge(df_weight.rename(columns = {'trddate':'weightdate'}),on = ['weightdate','code'],how = 'left')
        # 根据trddate和nxtrebalance为因子表匹配下一期收益率
        df_factor = self.matchret(df_factor)
        # 删掉计算时的冗余列
        df_factor = df_factor[['trddate','code','ret'] + self.weightlst + self.factornamelst].reset_index(drop = True)
        if ascending is not None:
            # 把非正向因子转为正向因子
            ascendingdict = dict(zip(self.factornamelst,list(ascending)))
            for factorname in ascendingdict:
                if not ascendingdict[factorname]:
                    df_factor[factorname] = -df_factor[factorname]
        # 输出
        return df_factor

    def matchrebalence(self,df_factor_it:pd.DataFrame,rebalancematch:pd.DataFrame)->pd.DataFrame:
        '''(聚合函数,用于apply)为资产在某个交易日上的数据匹配再平衡日期'''
        start,end = df_factor_it['trddate'].values[0],df_factor_it['nxtfactor'].values[0]
        factormatch = rebalancematch.loc[(rebalancematch['trddate'] > start)*(rebalancematch['trddate'] <= end),:]
        if len(factormatch) != 0:
            matchvalues = df_factor_it[['code'] + self.factornamelst].values[0]
            factormatch.loc[:,['code'] + self.factornamelst] = list(matchvalues)
            return factormatch
        else:
            return pd.DataFrame()

class SingleSort:
    def __init__(self,df_ic:pd.DataFrame,df_icir:pd.DataFrame,df_ratios:pd.DataFrame,factorcorr:pd.DataFrame,dict_id:Dict[str,pd.DataFrame],dict_ret:Dict[str,pd.DataFrame],dict_netval:Dict[str,pd.DataFrame],dict_fee:Dict[str,pd.DataFrame],dict_trdret:Dict[str,pd.DataFrame],dict_trdnetval:Dict[str,pd.DataFrame]):
        '''
        SingleSort类专用于存储和获取单分组结果,画图

        df_ic:因子IC,共5列
            第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
            第2列为factorname:str,因子名
            第3列为当期完整的股票和因子数据条数:int
            第4列为ic:float,因子当期ic
            第5列为rankic:float,因子当期rankic

        df_icir:因子ICIR,共5列
            第1列为factorname:str,因子名
            第2列为时序平均股票和因子数据条数:int
            第3列为avgic:float,因子时序平均ic
            第4列为avgrankic:float,因子时序平均rankic
            第5列为ir:float,因子在整个回测期内的ir
            第6列为rankir:float,因子在整个回测期内的rankir

        df_ratios:因子分组年化收益率,最大回撤率,收益率的newey-west t值,共9列
            第1列为factorname:str,因子名
            第2列为weight:str,组合加权方式
            第3列为id:str,1~g的分组结果以及'singlesort'
            第4列为annret:float,不考虑费用的组合年化收益率
            第5列为anntrdret:float,考虑费用的组合年化收益率
            第6列为retmaxdrawdown:float,不考虑费用的组合最大回撤率
            第7列为trdretmaxdrawdown:float,考虑费用的组合最大回撤率
            第8列为rettval:float,不考虑费用的组合收益率t值
            第9列为trdrettval:float,考虑费用的组合收益率t值

        factorcorr:因子相关系数表
            方阵,因子截面相关系数矩阵的均值

        dict_id:资产分组表
            keys:因子名
            values:df_factor,资产分组结果和归一化后的权重表
                第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
                第2列为code:str,资产代码
                第3列为id:str(int),1~g的分组结果以及'singlesort'
                第4~4+len(self.weightlst)列为weightlst:float,各组的组内资产权重

        dict_ret:组合收益率表
            keys:因子名
            values:df_group,分组收益率
                第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
                第2列为id:str(int),1~g的分组结果以及'singlesort'
                第3~3+len(self.weightlst)列为weightlst:float,各种加权方式的分组收益率

        dict_netval:考虑交易费用的组合净值表
            keys:因子名
            values:df_netval,考虑交易费用的组合净值表
                第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
                第2列为id:str(int),1~g的分组结果以及'singlesort'
                第3~3+len(self.weightlst)列为weightlst:float,各种加权方式的组合净值

        dict_fee:组合交易费用表
            keys:因子名
            values:df_fee,交易费用表
                第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
                第2列为id:str(int),1~g的分组结果以及'singlesort'
                第3~3+len(self.weightlst)列为weightlst:float,各种加权方式的交易费用

        dict_trdret:考虑交易费用的组合收益率表
            keys:因子名
            values:df_trdret,考虑交易费用的组合收益率表
                第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
                第2列为id:str(int),1~g的分组结果以及'singlesort'
                第3~3+len(self.weightlst)列为weightlst:float,各种加权方式的交易费用下的组合收益率
        
        dict_trdnetval:考虑交易费用的组合净值表
            keys:因子名
            values:df_trdnetval,考虑交易费用的组合净值表
                第1列为trddate:str,格式为'%Y%m%d'统一设置为再平衡交易日
                第2列为id:str(int),1~g的分组结果以及'singlesort'
                第3~3+len(self.weightlst)列为weightlst:float,各种加权方式的交易费用下的组合净值
        '''
        self.factornamelst = list(dict_id.keys())
        self.weightlst = list(dict_id[self.factornamelst[0]].columns[3:])
        self.df_ic = df_ic
        self.df_icir = df_icir
        self.df_ratios = df_ratios
        self.factorcorr = factorcorr
        self.dict_id = dict_id
        self.dict_ret = dict_ret
        self.dict_netval = dict_netval
        self.dict_fee = dict_fee
        self.dict_trdret = dict_trdret
        self.dict_trdnetval = dict_trdnetval
    
    def get_factorname(self)->List[str]:
        '''获取因子列表'''
        return self.factornamelst

    def get_ic(self)->pd.DataFrame:
        '''获取因子IC'''
        return self.df_ic

    def get_icir(self)->pd.DataFrame:
        '''获取因子ICIR'''
        return self.df_icir
    
    def get_ratios(self)->pd.DataFrame:
        '''获取组合年化收益率,最大回撤率,收益率t值'''
        return self.df_ratios

    def get_corr(self)->pd.DataFrame:
        '''获取因子截面相关系数矩阵的时序均值'''
        return self.factorcorr

    def get_id(self)->Dict[str,pd.DataFrame]:
        '''获取分组结果'''
        return self.dict_id

    def get_ret(self)->Dict[str,pd.DataFrame]:
        '''获取分组收益率'''
        return self.dict_ret

    def get_netval(self)->Dict[str,pd.DataFrame]:
        '''获取分组净值'''
        return self.dict_netval

    def get_fee(self)->Dict[str,pd.DataFrame]:
        '''获取组合交易费用'''
        return self.dict_fee

    def get_trdret(self)->Dict[str,pd.DataFrame]:
        '''获取考虑交易费用的分组收益率'''
        return self.dict_trdret

    def get_trdnetval(self)->Dict[str,pd.DataFrame]:
        '''获取考虑交易费用的分组净值'''
        return self.dict_trdnetval

    def plot_ret(self,path:str,considerfee:bool = False):
        '''
        画出分组累计净值
        
        参数:
    
            path:包含图片名的path

            considerfee:False时输出不考虑交易费用的累计净值,True时输出考虑交易费用的累计净值
        '''
        factornum,weightnum = len(self.factornamelst),len(self.weightlst)
        fig,axs = plt.subplots(weightnum,factornum,figsize = [factornum*10,weightnum*5])
        for i in range(factornum):
            factorname = self.factornamelst[i]
            df_netval = self.dict_netval[factorname].copy() if considerfee == False else self.dict_trdnetval[factorname].copy()
            weightlst = self.weightlst if considerfee == False else [f'trd{weightname}' for weightname in self.weightlst]
            df_netval.loc[:,'trddate'] = df_netval.loc[:,'trddate'].apply(lambda x: dt.datetime.strptime(x,'%Y%m%d').date())
            for j in range(weightnum):
                baseweightname = self.weightlst[j]
                weightname = weightlst[j]
                df_netval_ = df_netval.loc[:,['trddate','id',weightname]]
                idlst = list(df_netval_['id'].drop_duplicates().sort_values().values)
                ax = axs[j,i]
                for id in idlst:
                    df_netval_id = df_netval_.loc[df_netval_['id'] == id,['trddate',weightname]]
                    boolloc = (self.df_ratios['factorname'] == factorname)*(self.df_ratios['weight'] == baseweightname)*(self.df_ratios['id'] == id)
                    tval = self.df_ratios.loc[boolloc,'rettval' if considerfee == False else 'trdrettval'].values[0]
                    annret = self.df_ratios.loc[boolloc,'annret' if considerfee == False else 'anntrdret'].values[0]
                    groupname = id if id == 'longshort' else f'{id}st group'
                    linename = f'{groupname}-annual ret:{annret*100:.2f}%-tval:{tval:.2f}'
                    ax.plot(df_netval_id['trddate'],df_netval_id[weightname],label = linename)
                ax.legend(loc = "upper left")
                ax.set_title(f'factor {factorname}-{baseweightname} weighted',fontsize = 16)
        fig.suptitle(f'Portfolio Cumulative Net Value',fontsize = 32)
        plt.savefig(path)
        plt.close('all')

    def describe(self,path:str):
        '''输出单分组的全部结果,输出到excel中,path是包含excel名的path'''
        print('saving singlesort results...')
        timestart = time()
        with pd.ExcelWriter(path,engine = 'openpyxl') as writer:
            # IC
            self.df_ic.to_excel(writer,sheet_name = 'IC',index = False)
            # ICIR
            self.df_icir.to_excel(writer,sheet_name = 'ICIR',index = False)
            # t值
            self.df_ratios.to_excel(writer,sheet_name = 'ratios',index = False)
            # 相关系数
            self.factorcorr.to_excel(writer,sheet_name = 'corr',index = False)
            for factorname in self.factornamelst:
                # 分组结果
                self.dict_id[factorname].to_excel(writer,sheet_name = f'{factorname}分组结果',index = False)
                # 分组收益率
                self.dict_ret[factorname].to_excel(writer,sheet_name = f'{factorname}收益率',index = False)
                # 考虑交易费用的分组收益率
                self.dict_trdret[factorname].to_excel(writer,sheet_name = f'{factorname}交易收益率',index = False)
        print(f'singlesort results saved, {time() - timestart:.2f}s passed.')