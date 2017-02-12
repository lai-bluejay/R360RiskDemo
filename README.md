[toc]
# readme文档要做什么:
+ 软件定位，软件的基本功能。
+ 运行代码的方法: 安装环境、启动命令等。
+ 简要的使用说明。
+ 代码目录结构说明，更详细点可以说明软件的基本原理。
+ 常见问题说明。

# 如何对待文档:
目录结构中有设docs/这个目录，用于存放代码文档。

在普通的项目中，确实没必要写非常详细的文档，可以考虑另一种风格: "在代码中写文档"。即在写代码的时候，在代码文件里把软件/模块的简要用法写明。简单有用。

# 1. 如何生成你的docs
长期有效的方式, 是用sphinx搭建你的文档目录.

## 1.1 如何使用Sphinx.
- 选择你要新建文档的地方, 如 docs/ , 执行:

```shell
$ sphinx-quickstart
```

注意填写项目的文档版本和项目名称, 在project language上, 选择 zh_CN

# 2.代码结构
## 2.1 概述
沿袭weflask框架, 并做微调.

新特征文件生成过程

在2016年12月30日星期五更新后的stat_model_v2代码中，重点优化了查询用户的sql代码。
其中，在mysql数据库中，同一用户存在不同状态的数条记录。例如，该用户之前被拒绝了，再次申请后又通过了，因此需要查找数据库中最新的用户状态STATUS和label。

涉及更新SQL语句的文件有：get_user_label.py, o2o_user_dao.py, feature_extractor.py等

1. 用label/get_user_label.py生成用户uid和label文件
涉及的更新有sql语句，筛选最新status和加入筛选时间begin, end

2. 用feature/feature_pre_generator/feature_i_gotcha.py生成特征和label文件
涉及的更新有代码的拆分
需要注意这里的时间begain, end需要与生成label的代码中涉及的文件相同

3. 用feature/feature_extraction/feature_extractor.py里的get_corr_with_label函数去检验特征与label之间的相关性

这里涉及的更新有，将相关性文件以csv格式输出
需要重点检查相关性绝对值在0.1以上的特征
•	是否是hit_black，4828，bank相关特征
•	是否用到了未来的时间维度
•	是不是跟用户的真实label相关

4. 训练数据准备完毕，将数据放入xgb模型中训练

新一版的o2o_xgb_ldc.py中涉及的更新有：
•	加入了l2 lamba, l1 alpha正则项系数，目前取值1e-5
•	加入了subsample, column subsample，目前两个参数都是0.8
•	在输出特征重要程度的函数record_feature_importance中，加入参数feature_type, 即将due&overdue model和add reject inference之后的model的feature importance分开
•	加入按Infomraiton gain对特征排序的代码(from kaggle)，代码加在了两个模型训练后dump_model之后，计划按两个model各自输出的feature做feature selection

新特征模型效果检验
对比模型：原始特征+原始xgb model(no l1, l2, subsample)
