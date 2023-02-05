# Version 2.04

#### BaseTransform

基础。

#### Compose

连续转换工具

#### ToDense

转换adj为 edge_index

#### ToSparseTensor

转换edge_index为 adj

#### ToDevice

设备转换

#### AddSelfLoops

在edge_weight 添加 节点自身的权重。

#### Cartesian LocalCartesian

相对笛卡尔坐标（x,y,z）保存到 edge_attr

#### Center

pos坐标 中心化

#### Constant

x 添加统一常数。

####Delaunay
face 狄洛尼三角剖分

#### Distance

把欧式距离添加到 edge_attr

#### FaceToEdge

face 转换为 edge_attr

#### GenerateMeshNormals

根据 face 生成法向量 保存到 norm

#### FixedPoints

从 node 中采样为 固定个数

#### GridSampling

网格采样，生成新node,edge ....

#### LineGraph

将图转换为线性图

#### NormalizeFeatures

所有属性归一化

#### NormalizeScale

pos的 center + normal

#### PointPairFeatures

计算旋转不变的点对特征加到 edge_attr

#### Polar

极坐标添加到 edge_attr

#### Spherical

球坐标添加到 edge_attr

#### SVDFeatureReduction

x 特征降维

#### RadiusGraph

根据pos 创建 edge_weight

#### RandomFlip

随机反转节点位置（按轴）

#### RandomFlip

链接度加入 edge_attr

#### NormalizeRotation

旋转pos?
Rotates all points according to the eigenvectors of the point cloud

#### AddMetapaths**

异构图。

#### LinearTransformation

对 pos 进行线性转换： pos 乘以（D，D）矩阵

#### RandomRotate

随机旋转 node 位置 pos

#### RandomScale

随机缩放 pos

#### RandomShear

随机剪切node

#### RandomTranslate

随即平移

#### RemoveIsolatedNodes

删除独立node

#### RemoveTrainingClasses

移除训练标签

#### SamplePoints

均匀采样

#### LaplacianLambdaMax

计算图拉普拉斯算子的最高特征值

#### LargestConnectedComponents

获取最大联通量的子图

#### GCNNorm

图卷积网络转换

#### KNNGraph

KNN 转换

#### GDC

通过图扩散卷积(GDC)对图进行处理
扩散提高了图表学习

##### LocalDegreeProfile

附加本地度配置文件(LDP)从“一个简单而有效的基线为非属性图分类

#### OneHotDegree

将节点度当作一个特征加入到x

#### RandomLinkSplit

执行edge级随机划分到培训，验证和测试(不适用)

#### RandomnodeSplit

执行node级随机划分到培训，验证和测试(不适用)

#### ToSLIC

图片转换为拓扑

#### ToUndirected

将同构或异构图转换为无向图

#### TwoHop

将跳点边加到边索引。

#### VirtualNode

添加虚拟节点

