са
ёЦ
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Р
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%ЭЬL>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Ѕ
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8э
|
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
d* 
shared_namedense_25/kernel
u
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel* 
_output_shapes
:
d*
dtype0
t
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_25/bias
m
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes

:*
dtype0

embedding_10/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*(
shared_nameembedding_10/embeddings

+embedding_10/embeddings/Read/ReadVariableOpReadVariableOpembedding_10/embeddings*
_output_shapes

:2*
dtype0
{
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2А* 
shared_namedense_26/kernel
t
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes
:	2А*
dtype0
s
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_26/bias
l
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes	
:А*
dtype0

conv2d_transpose_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_15/kernel

.conv2d_transpose_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_15/kernel*(
_output_shapes
:*
dtype0

conv2d_transpose_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_15/bias

,conv2d_transpose_15/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_15/bias*
_output_shapes	
:*
dtype0

conv2d_transpose_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_16/kernel

.conv2d_transpose_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_16/kernel*(
_output_shapes
:*
dtype0

conv2d_transpose_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_16/bias

,conv2d_transpose_16/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_16/bias*
_output_shapes	
:*
dtype0

conv2d_transpose_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_17/kernel

.conv2d_transpose_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_17/kernel*(
_output_shapes
:*
dtype0

conv2d_transpose_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_17/bias

,conv2d_transpose_17/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_17/bias*
_output_shapes	
:*
dtype0

conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_15/kernel
~
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*'
_output_shapes
:*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Э3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*3
valueў2Bћ2 Bє2
§
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer_with_weights-5
layer-13
layer-14
layer_with_weights-6
layer-15
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
b

embeddings
trainable_variables
regularization_losses
 	variables
!	keras_api
R
"trainable_variables
#regularization_losses
$	variables
%	keras_api
h

&kernel
'bias
(trainable_variables
)regularization_losses
*	variables
+	keras_api
R
,trainable_variables
-regularization_losses
.	variables
/	keras_api
R
0trainable_variables
1regularization_losses
2	variables
3	keras_api
R
4trainable_variables
5regularization_losses
6	variables
7	keras_api
h

8kernel
9bias
:trainable_variables
;regularization_losses
<	variables
=	keras_api
R
>trainable_variables
?regularization_losses
@	variables
A	keras_api
h

Bkernel
Cbias
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
R
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
h

Lkernel
Mbias
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
R
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
h

Vkernel
Wbias
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
 
^
0
1
2
&3
'4
85
96
B7
C8
L9
M10
V11
W12
 
^
0
1
2
&3
'4
85
96
B7
C8
L9
M10
V11
W12
­
\layer_metrics

]layers
^metrics
_non_trainable_variables
`layer_regularization_losses
trainable_variables
regularization_losses
	variables
 
[Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_25/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
alayer_metrics

blayers
cmetrics
dnon_trainable_variables
elayer_regularization_losses
trainable_variables
regularization_losses
	variables
ge
VARIABLE_VALUEembedding_10/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
­
flayer_metrics

glayers
hmetrics
inon_trainable_variables
jlayer_regularization_losses
trainable_variables
regularization_losses
 	variables
 
 
 
­
klayer_metrics

llayers
mmetrics
nnon_trainable_variables
olayer_regularization_losses
"trainable_variables
#regularization_losses
$	variables
[Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_26/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1
 

&0
'1
­
player_metrics

qlayers
rmetrics
snon_trainable_variables
tlayer_regularization_losses
(trainable_variables
)regularization_losses
*	variables
 
 
 
­
ulayer_metrics

vlayers
wmetrics
xnon_trainable_variables
ylayer_regularization_losses
,trainable_variables
-regularization_losses
.	variables
 
 
 
­
zlayer_metrics

{layers
|metrics
}non_trainable_variables
~layer_regularization_losses
0trainable_variables
1regularization_losses
2	variables
 
 
 
Б
layer_metrics
layers
metrics
non_trainable_variables
 layer_regularization_losses
4trainable_variables
5regularization_losses
6	variables
fd
VARIABLE_VALUEconv2d_transpose_15/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_15/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91
 

80
91
В
layer_metrics
layers
metrics
non_trainable_variables
 layer_regularization_losses
:trainable_variables
;regularization_losses
<	variables
 
 
 
В
layer_metrics
layers
metrics
non_trainable_variables
 layer_regularization_losses
>trainable_variables
?regularization_losses
@	variables
fd
VARIABLE_VALUEconv2d_transpose_16/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_16/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
 

B0
C1
В
layer_metrics
layers
metrics
non_trainable_variables
 layer_regularization_losses
Dtrainable_variables
Eregularization_losses
F	variables
 
 
 
В
layer_metrics
layers
metrics
non_trainable_variables
 layer_regularization_losses
Htrainable_variables
Iregularization_losses
J	variables
fd
VARIABLE_VALUEconv2d_transpose_17/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_17/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1
 

L0
M1
В
layer_metrics
layers
metrics
non_trainable_variables
 layer_regularization_losses
Ntrainable_variables
Oregularization_losses
P	variables
 
 
 
В
layer_metrics
layers
metrics
 non_trainable_variables
 Ёlayer_regularization_losses
Rtrainable_variables
Sregularization_losses
T	variables
\Z
VARIABLE_VALUEconv2d_15/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_15/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

V0
W1
 

V0
W1
В
Ђlayer_metrics
Ѓlayers
Єmetrics
Ѕnon_trainable_variables
 Іlayer_regularization_losses
Xtrainable_variables
Yregularization_losses
Z	variables
 
v
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
{
serving_default_input_21Placeholder*'
_output_shapes
:џџџџџџџџџd*
dtype0*
shape:џџџџџџџџџd
{
serving_default_input_22Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_21serving_default_input_22embedding_10/embeddingsdense_25/kerneldense_25/biasdense_26/kerneldense_26/biasconv2d_transpose_15/kernelconv2d_transpose_15/biasconv2d_transpose_16/kernelconv2d_transpose_16/biasconv2d_transpose_17/kernelconv2d_transpose_17/biasconv2d_15/kernelconv2d_15/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ``*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_15572760
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ю
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp+embedding_10/embeddings/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp.conv2d_transpose_15/kernel/Read/ReadVariableOp,conv2d_transpose_15/bias/Read/ReadVariableOp.conv2d_transpose_16/kernel/Read/ReadVariableOp,conv2d_transpose_16/bias/Read/ReadVariableOp.conv2d_transpose_17/kernel/Read/ReadVariableOp,conv2d_transpose_17/bias/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_save_15573339
Х
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_25/kerneldense_25/biasembedding_10/embeddingsdense_26/kerneldense_26/biasconv2d_transpose_15/kernelconv2d_transpose_15/biasconv2d_transpose_16/kernelconv2d_transpose_16/biasconv2d_transpose_17/kernelconv2d_transpose_17/biasconv2d_15/kernelconv2d_15/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference__traced_restore_15573388Њ
ц

6__inference_conv2d_transpose_15_layer_call_fn_15572184

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_155721742
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ё(

!__inference__traced_save_15573339
file_prefix.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop6
2savev2_embedding_10_embeddings_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop9
5savev2_conv2d_transpose_15_kernel_read_readvariableop7
3savev2_conv2d_transpose_15_bias_read_readvariableop9
5savev2_conv2d_transpose_16_kernel_read_readvariableop7
3savev2_conv2d_transpose_16_bias_read_readvariableop9
5savev2_conv2d_transpose_17_kernel_read_readvariableop7
3savev2_conv2d_transpose_17_bias_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameё
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueљBіB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЄ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesУ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop2savev2_embedding_10_embeddings_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop5savev2_conv2d_transpose_15_kernel_read_readvariableop3savev2_conv2d_transpose_15_bias_read_readvariableop5savev2_conv2d_transpose_16_kernel_read_readvariableop3savev2_conv2d_transpose_16_bias_read_readvariableop5savev2_conv2d_transpose_17_kernel_read_readvariableop3savev2_conv2d_transpose_17_bias_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Г
_input_shapesЁ
: :
d::2:	2А:А::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
d:"

_output_shapes

::$ 

_output_shapes

:2:%!

_output_shapes
:	2А:!

_output_shapes	
:А:.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!	

_output_shapes	
::.
*
(
_output_shapes
::!

_output_shapes	
::-)
'
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 

З
+__inference_model_13_layer_call_fn_15572649
input_21
input_22
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identityЂStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinput_21input_22unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_model_13_layer_call_and_return_conditional_losses_155726202
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:џџџџџџџџџd:џџџџџџџџџ:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
input_21:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_22
ѕ
x
L__inference_concatenate_10_layer_call_and_return_conditional_losses_15573220
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:џџџџџџџџџ2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ:џџџџџџџџџ:Z V
0
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
Э
h
L__inference_leaky_re_lu_35_layer_call_and_return_conditional_losses_15572376

inputs
identityV
	LeakyRelu	LeakyReluinputs*)
_output_shapes
:џџџџџџџџџ2
	LeakyRelum
IdentityIdentityLeakyRelu:activations:0*
T0*)
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ:Q M
)
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к
х
F__inference_dense_26_layer_call_and_return_conditional_losses_15572355

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	2А*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ22
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџА2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџА2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:џџџџџџџџџА2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
ј#
ў
Q__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_15572218

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :2	
stack/3
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ь
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3Е
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpё
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddК
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Г
I
-__inference_reshape_15_layer_call_fn_15573194

inputs
identityв
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_15_layer_call_and_return_conditional_losses_155723982
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ:Q M
)
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Н:
І
$__inference__traced_restore_15573388
file_prefix$
 assignvariableop_dense_25_kernel$
 assignvariableop_1_dense_25_bias.
*assignvariableop_2_embedding_10_embeddings&
"assignvariableop_3_dense_26_kernel$
 assignvariableop_4_dense_26_bias1
-assignvariableop_5_conv2d_transpose_15_kernel/
+assignvariableop_6_conv2d_transpose_15_bias1
-assignvariableop_7_conv2d_transpose_16_kernel/
+assignvariableop_8_conv2d_transpose_16_bias1
-assignvariableop_9_conv2d_transpose_17_kernel0
,assignvariableop_10_conv2d_transpose_17_bias(
$assignvariableop_11_conv2d_15_kernel&
"assignvariableop_12_conv2d_15_bias
identity_14ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9ї
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueљBіB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЊ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesё
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_25_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѕ
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_25_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Џ
AssignVariableOp_2AssignVariableOp*assignvariableop_2_embedding_10_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ї
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_26_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ѕ
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_26_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5В
AssignVariableOp_5AssignVariableOp-assignvariableop_5_conv2d_transpose_15_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6А
AssignVariableOp_6AssignVariableOp+assignvariableop_6_conv2d_transpose_15_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7В
AssignVariableOp_7AssignVariableOp-assignvariableop_7_conv2d_transpose_16_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8А
AssignVariableOp_8AssignVariableOp+assignvariableop_8_conv2d_transpose_16_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9В
AssignVariableOp_9AssignVariableOp-assignvariableop_9_conv2d_transpose_17_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Д
AssignVariableOp_10AssignVariableOp,assignvariableop_10_conv2d_transpose_17_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ќ
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_15_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Њ
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_15_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpќ
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_13я
Identity_14IdentityIdentity_13:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_14"#
identity_14Identity_14:output:0*I
_input_shapes8
6: :::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
а

,__inference_conv2d_15_layer_call_fn_15573276

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_15_layer_call_and_return_conditional_losses_155725092
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
В
h
L__inference_leaky_re_lu_38_layer_call_and_return_conditional_losses_15573251

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

З
+__inference_model_13_layer_call_fn_15573090
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identityЂStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_model_13_layer_call_and_return_conditional_losses_155726972
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:џџџџџџџџџd:џџџџџџџџџ:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
і

+__inference_dense_26_layer_call_fn_15573175

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_155723552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџА2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ2::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
№
d
H__inference_reshape_15_layer_call_and_return_conditional_losses_15573189

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape/shape/3К
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ:Q M
)
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
г
u
/__inference_embedding_10_layer_call_fn_15573126

inputs
unknown
identityЂStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_embedding_10_layer_call_and_return_conditional_losses_155722872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
В
h
L__inference_leaky_re_lu_37_layer_call_and_return_conditional_losses_15573241

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
гA

F__inference_model_13_layer_call_and_return_conditional_losses_15572697

inputs
inputs_1
embedding_10_15572656
dense_25_15572659
dense_25_15572661
dense_26_15572664
dense_26_15572666 
conv2d_transpose_15_15572673 
conv2d_transpose_15_15572675 
conv2d_transpose_16_15572679 
conv2d_transpose_16_15572681 
conv2d_transpose_17_15572685 
conv2d_transpose_17_15572687
conv2d_15_15572691
conv2d_15_15572693
identityЂ!conv2d_15/StatefulPartitionedCallЂ+conv2d_transpose_15/StatefulPartitionedCallЂ+conv2d_transpose_16/StatefulPartitionedCallЂ+conv2d_transpose_17/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ dense_26/StatefulPartitionedCallЂ$embedding_10/StatefulPartitionedCall
$embedding_10/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_10_15572656*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_embedding_10_layer_call_and_return_conditional_losses_155722872&
$embedding_10/StatefulPartitionedCall
 dense_25/StatefulPartitionedCallStatefulPartitionedCallinputsdense_25_15572659dense_25_15572661*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_155723092"
 dense_25/StatefulPartitionedCallЩ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall-embedding_10/StatefulPartitionedCall:output:0dense_26_15572664dense_26_15572666*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_155723552"
 dense_26/StatefulPartitionedCall
leaky_re_lu_35/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_35_layer_call_and_return_conditional_losses_155723762 
leaky_re_lu_35/PartitionedCall
reshape_15/PartitionedCallPartitionedCall'leaky_re_lu_35/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_15_layer_call_and_return_conditional_losses_155723982
reshape_15/PartitionedCall
reshape_16/PartitionedCallPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_16_layer_call_and_return_conditional_losses_155724202
reshape_16/PartitionedCallЗ
concatenate_10/PartitionedCallPartitionedCall#reshape_15/PartitionedCall:output:0#reshape_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_concatenate_10_layer_call_and_return_conditional_losses_155724352 
concatenate_10/PartitionedCall
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0conv2d_transpose_15_15572673conv2d_transpose_15_15572675*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_155721742-
+conv2d_transpose_15/StatefulPartitionedCallД
leaky_re_lu_36/PartitionedCallPartitionedCall4conv2d_transpose_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_36_layer_call_and_return_conditional_losses_155724542 
leaky_re_lu_36/PartitionedCall
+conv2d_transpose_16/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_36/PartitionedCall:output:0conv2d_transpose_16_15572679conv2d_transpose_16_15572681*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_155722182-
+conv2d_transpose_16/StatefulPartitionedCallД
leaky_re_lu_37/PartitionedCallPartitionedCall4conv2d_transpose_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_37_layer_call_and_return_conditional_losses_155724722 
leaky_re_lu_37/PartitionedCall
+conv2d_transpose_17/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_37/PartitionedCall:output:0conv2d_transpose_17_15572685conv2d_transpose_17_15572687*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_155722622-
+conv2d_transpose_17/StatefulPartitionedCallД
leaky_re_lu_38/PartitionedCallPartitionedCall4conv2d_transpose_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_38_layer_call_and_return_conditional_losses_155724902 
leaky_re_lu_38/PartitionedCallн
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_38/PartitionedCall:output:0conv2d_15_15572691conv2d_15_15572693*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_15_layer_call_and_return_conditional_losses_155725092#
!conv2d_15/StatefulPartitionedCallГ
IdentityIdentity*conv2d_15/StatefulPartitionedCall:output:0"^conv2d_15/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall,^conv2d_transpose_16/StatefulPartitionedCall,^conv2d_transpose_17/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall%^embedding_10/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:џџџџџџџџџd:џџџџџџџџџ:::::::::::::2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2Z
+conv2d_transpose_16/StatefulPartitionedCall+conv2d_transpose_16/StatefulPartitionedCall2Z
+conv2d_transpose_17/StatefulPartitionedCall+conv2d_transpose_17/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2L
$embedding_10/StatefulPartitionedCall$embedding_10/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

M
1__inference_leaky_re_lu_38_layer_call_fn_15573256

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_38_layer_call_and_return_conditional_losses_155724902
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ѓ
d
H__inference_reshape_16_layer_call_and_return_conditional_losses_15572420

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3К
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџА:T P
,
_output_shapes
:џџџџџџџџџА
 
_user_specified_nameinputs
	
п
F__inference_dense_25_layer_call_and_return_conditional_losses_15573100

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
d*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*)
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
ц

6__inference_conv2d_transpose_17_layer_call_fn_15572272

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_155722622
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
№
d
H__inference_reshape_15_layer_call_and_return_conditional_losses_15572398

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape/shape/3К
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ:Q M
)
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
В
h
L__inference_leaky_re_lu_37_layer_call_and_return_conditional_losses_15572472

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
п
F__inference_dense_25_layer_call_and_return_conditional_losses_15572309

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
d*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*)
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ж
р
G__inference_conv2d_15_layer_call_and_return_conditional_losses_15573267

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
TanhЇ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ж
р
G__inference_conv2d_15_layer_call_and_return_conditional_losses_15572509

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
TanhЇ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
бС
П

F__inference_model_13_layer_call_and_return_conditional_losses_15572893
inputs_0
inputs_1*
&embedding_10_embedding_lookup_15572765+
'dense_25_matmul_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource.
*dense_26_tensordot_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource@
<conv2d_transpose_15_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_15_biasadd_readvariableop_resource@
<conv2d_transpose_16_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_16_biasadd_readvariableop_resource@
<conv2d_transpose_17_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_17_biasadd_readvariableop_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource
identityЂ conv2d_15/BiasAdd/ReadVariableOpЂconv2d_15/Conv2D/ReadVariableOpЂ*conv2d_transpose_15/BiasAdd/ReadVariableOpЂ3conv2d_transpose_15/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_16/BiasAdd/ReadVariableOpЂ3conv2d_transpose_16/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_17/BiasAdd/ReadVariableOpЂ3conv2d_transpose_17/conv2d_transpose/ReadVariableOpЂdense_25/BiasAdd/ReadVariableOpЂdense_25/MatMul/ReadVariableOpЂdense_26/BiasAdd/ReadVariableOpЂ!dense_26/Tensordot/ReadVariableOpЂembedding_10/embedding_lookupy
embedding_10/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
embedding_10/CastТ
embedding_10/embedding_lookupResourceGather&embedding_10_embedding_lookup_15572765embedding_10/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_10/embedding_lookup/15572765*+
_output_shapes
:џџџџџџџџџ2*
dtype02
embedding_10/embedding_lookupЃ
&embedding_10/embedding_lookup/IdentityIdentity&embedding_10/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_10/embedding_lookup/15572765*+
_output_shapes
:џџџџџџџџџ22(
&embedding_10/embedding_lookup/IdentityЧ
(embedding_10/embedding_lookup/Identity_1Identity/embedding_10/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ22*
(embedding_10/embedding_lookup/Identity_1Њ
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype02 
dense_25/MatMul/ReadVariableOp
dense_25/MatMulMatMulinputs_0&dense_25/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:џџџџџџџџџ2
dense_25/MatMulЉ
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_25/BiasAdd/ReadVariableOpЇ
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:џџџџџџџџџ2
dense_25/BiasAddВ
!dense_26/Tensordot/ReadVariableOpReadVariableOp*dense_26_tensordot_readvariableop_resource*
_output_shapes
:	2А*
dtype02#
!dense_26/Tensordot/ReadVariableOp|
dense_26/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_26/Tensordot/axes
dense_26/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_26/Tensordot/free
dense_26/Tensordot/ShapeShape1embedding_10/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense_26/Tensordot/Shape
 dense_26/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_26/Tensordot/GatherV2/axisў
dense_26/Tensordot/GatherV2GatherV2!dense_26/Tensordot/Shape:output:0 dense_26/Tensordot/free:output:0)dense_26/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_26/Tensordot/GatherV2
"dense_26/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_26/Tensordot/GatherV2_1/axis
dense_26/Tensordot/GatherV2_1GatherV2!dense_26/Tensordot/Shape:output:0 dense_26/Tensordot/axes:output:0+dense_26/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_26/Tensordot/GatherV2_1~
dense_26/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_26/Tensordot/ConstЄ
dense_26/Tensordot/ProdProd$dense_26/Tensordot/GatherV2:output:0!dense_26/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_26/Tensordot/Prod
dense_26/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_26/Tensordot/Const_1Ќ
dense_26/Tensordot/Prod_1Prod&dense_26/Tensordot/GatherV2_1:output:0#dense_26/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_26/Tensordot/Prod_1
dense_26/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_26/Tensordot/concat/axisн
dense_26/Tensordot/concatConcatV2 dense_26/Tensordot/free:output:0 dense_26/Tensordot/axes:output:0'dense_26/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_26/Tensordot/concatА
dense_26/Tensordot/stackPack dense_26/Tensordot/Prod:output:0"dense_26/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_26/Tensordot/stackж
dense_26/Tensordot/transpose	Transpose1embedding_10/embedding_lookup/Identity_1:output:0"dense_26/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ22
dense_26/Tensordot/transposeУ
dense_26/Tensordot/ReshapeReshape dense_26/Tensordot/transpose:y:0!dense_26/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_26/Tensordot/ReshapeУ
dense_26/Tensordot/MatMulMatMul#dense_26/Tensordot/Reshape:output:0)dense_26/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА2
dense_26/Tensordot/MatMul
dense_26/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
dense_26/Tensordot/Const_2
 dense_26/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_26/Tensordot/concat_1/axisъ
dense_26/Tensordot/concat_1ConcatV2$dense_26/Tensordot/GatherV2:output:0#dense_26/Tensordot/Const_2:output:0)dense_26/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_26/Tensordot/concat_1Е
dense_26/TensordotReshape#dense_26/Tensordot/MatMul:product:0$dense_26/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџА2
dense_26/TensordotЈ
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_26/BiasAdd/ReadVariableOpЌ
dense_26/BiasAddBiasAdddense_26/Tensordot:output:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџА2
dense_26/BiasAdd
leaky_re_lu_35/LeakyRelu	LeakyReludense_25/BiasAdd:output:0*)
_output_shapes
:џџџџџџџџџ2
leaky_re_lu_35/LeakyReluz
reshape_15/ShapeShape&leaky_re_lu_35/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_15/Shape
reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_15/strided_slice/stack
 reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_1
 reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_2Є
reshape_15/strided_sliceStridedSlicereshape_15/Shape:output:0'reshape_15/strided_slice/stack:output:0)reshape_15/strided_slice/stack_1:output:0)reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_15/strided_slicez
reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_15/Reshape/shape/1z
reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_15/Reshape/shape/2{
reshape_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
reshape_15/Reshape/shape/3ќ
reshape_15/Reshape/shapePack!reshape_15/strided_slice:output:0#reshape_15/Reshape/shape/1:output:0#reshape_15/Reshape/shape/2:output:0#reshape_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_15/Reshape/shapeЙ
reshape_15/ReshapeReshape&leaky_re_lu_35/LeakyRelu:activations:0!reshape_15/Reshape/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
reshape_15/Reshapem
reshape_16/ShapeShapedense_26/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_16/Shape
reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_16/strided_slice/stack
 reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_1
 reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_2Є
reshape_16/strided_sliceStridedSlicereshape_16/Shape:output:0'reshape_16/strided_slice/stack:output:0)reshape_16/strided_slice/stack_1:output:0)reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_16/strided_slicez
reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/1z
reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/2z
reshape_16/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/3ќ
reshape_16/Reshape/shapePack!reshape_16/strided_slice:output:0#reshape_16/Reshape/shape/1:output:0#reshape_16/Reshape/shape/2:output:0#reshape_16/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_16/Reshape/shapeЋ
reshape_16/ReshapeReshapedense_26/BiasAdd:output:0!reshape_16/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
reshape_16/Reshapez
concatenate_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_10/concat/axisн
concatenate_10/concatConcatV2reshape_15/Reshape:output:0reshape_16/Reshape:output:0#concatenate_10/concat/axis:output:0*
N*
T0*0
_output_shapes
:џџџџџџџџџ2
concatenate_10/concat
conv2d_transpose_15/ShapeShapeconcatenate_10/concat:output:0*
T0*
_output_shapes
:2
conv2d_transpose_15/Shape
'conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_15/strided_slice/stack 
)conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_15/strided_slice/stack_1 
)conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_15/strided_slice/stack_2к
!conv2d_transpose_15/strided_sliceStridedSlice"conv2d_transpose_15/Shape:output:00conv2d_transpose_15/strided_slice/stack:output:02conv2d_transpose_15/strided_slice/stack_1:output:02conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_15/strided_slice|
conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_15/stack/1|
conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_15/stack/2}
conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value
B :2
conv2d_transpose_15/stack/3
conv2d_transpose_15/stackPack*conv2d_transpose_15/strided_slice:output:0$conv2d_transpose_15/stack/1:output:0$conv2d_transpose_15/stack/2:output:0$conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_15/stack 
)conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_15/strided_slice_1/stackЄ
+conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_15/strided_slice_1/stack_1Є
+conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_15/strided_slice_1/stack_2ф
#conv2d_transpose_15/strided_slice_1StridedSlice"conv2d_transpose_15/stack:output:02conv2d_transpose_15/strided_slice_1/stack:output:04conv2d_transpose_15/strided_slice_1/stack_1:output:04conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_15/strided_slice_1ё
3conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_15_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype025
3conv2d_transpose_15/conv2d_transpose/ReadVariableOpЧ
$conv2d_transpose_15/conv2d_transposeConv2DBackpropInput"conv2d_transpose_15/stack:output:0;conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0concatenate_10/concat:output:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2&
$conv2d_transpose_15/conv2d_transposeЩ
*conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*conv2d_transpose_15/BiasAdd/ReadVariableOpу
conv2d_transpose_15/BiasAddBiasAdd-conv2d_transpose_15/conv2d_transpose:output:02conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_transpose_15/BiasAdd
leaky_re_lu_36/LeakyRelu	LeakyRelu$conv2d_transpose_15/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ2
leaky_re_lu_36/LeakyRelu
conv2d_transpose_16/ShapeShape&leaky_re_lu_36/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_16/Shape
'conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_16/strided_slice/stack 
)conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_16/strided_slice/stack_1 
)conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_16/strided_slice/stack_2к
!conv2d_transpose_16/strided_sliceStridedSlice"conv2d_transpose_16/Shape:output:00conv2d_transpose_16/strided_slice/stack:output:02conv2d_transpose_16/strided_slice/stack_1:output:02conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_16/strided_slice|
conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :02
conv2d_transpose_16/stack/1|
conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :02
conv2d_transpose_16/stack/2}
conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value
B :2
conv2d_transpose_16/stack/3
conv2d_transpose_16/stackPack*conv2d_transpose_16/strided_slice:output:0$conv2d_transpose_16/stack/1:output:0$conv2d_transpose_16/stack/2:output:0$conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_16/stack 
)conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_16/strided_slice_1/stackЄ
+conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_16/strided_slice_1/stack_1Є
+conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_16/strided_slice_1/stack_2ф
#conv2d_transpose_16/strided_slice_1StridedSlice"conv2d_transpose_16/stack:output:02conv2d_transpose_16/strided_slice_1/stack:output:04conv2d_transpose_16/strided_slice_1/stack_1:output:04conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_16/strided_slice_1ё
3conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_16_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype025
3conv2d_transpose_16/conv2d_transpose/ReadVariableOpЯ
$conv2d_transpose_16/conv2d_transposeConv2DBackpropInput"conv2d_transpose_16/stack:output:0;conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_36/LeakyRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ00*
paddingSAME*
strides
2&
$conv2d_transpose_16/conv2d_transposeЩ
*conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*conv2d_transpose_16/BiasAdd/ReadVariableOpу
conv2d_transpose_16/BiasAddBiasAdd-conv2d_transpose_16/conv2d_transpose:output:02conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ002
conv2d_transpose_16/BiasAdd
leaky_re_lu_37/LeakyRelu	LeakyRelu$conv2d_transpose_16/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ002
leaky_re_lu_37/LeakyRelu
conv2d_transpose_17/ShapeShape&leaky_re_lu_37/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_17/Shape
'conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_17/strided_slice/stack 
)conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_17/strided_slice/stack_1 
)conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_17/strided_slice/stack_2к
!conv2d_transpose_17/strided_sliceStridedSlice"conv2d_transpose_17/Shape:output:00conv2d_transpose_17/strided_slice/stack:output:02conv2d_transpose_17/strided_slice/stack_1:output:02conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_17/strided_slice|
conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`2
conv2d_transpose_17/stack/1|
conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value	B :`2
conv2d_transpose_17/stack/2}
conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value
B :2
conv2d_transpose_17/stack/3
conv2d_transpose_17/stackPack*conv2d_transpose_17/strided_slice:output:0$conv2d_transpose_17/stack/1:output:0$conv2d_transpose_17/stack/2:output:0$conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_17/stack 
)conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_17/strided_slice_1/stackЄ
+conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_17/strided_slice_1/stack_1Є
+conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_17/strided_slice_1/stack_2ф
#conv2d_transpose_17/strided_slice_1StridedSlice"conv2d_transpose_17/stack:output:02conv2d_transpose_17/strided_slice_1/stack:output:04conv2d_transpose_17/strided_slice_1/stack_1:output:04conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_17/strided_slice_1ё
3conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_17_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype025
3conv2d_transpose_17/conv2d_transpose/ReadVariableOpЯ
$conv2d_transpose_17/conv2d_transposeConv2DBackpropInput"conv2d_transpose_17/stack:output:0;conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_37/LeakyRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ``*
paddingSAME*
strides
2&
$conv2d_transpose_17/conv2d_transposeЩ
*conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*conv2d_transpose_17/BiasAdd/ReadVariableOpу
conv2d_transpose_17/BiasAddBiasAdd-conv2d_transpose_17/conv2d_transpose:output:02conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ``2
conv2d_transpose_17/BiasAdd
leaky_re_lu_38/LeakyRelu	LeakyRelu$conv2d_transpose_17/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ``2
leaky_re_lu_38/LeakyReluД
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_15/Conv2D/ReadVariableOpс
conv2d_15/Conv2DConv2D&leaky_re_lu_38/LeakyRelu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ``*
paddingSAME*
strides
2
conv2d_15/Conv2DЊ
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOpА
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ``2
conv2d_15/BiasAdd~
conv2d_15/TanhTanhconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ``2
conv2d_15/Tanh
IdentityIdentityconv2d_15/Tanh:y:0!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp+^conv2d_transpose_15/BiasAdd/ReadVariableOp4^conv2d_transpose_15/conv2d_transpose/ReadVariableOp+^conv2d_transpose_16/BiasAdd/ReadVariableOp4^conv2d_transpose_16/conv2d_transpose/ReadVariableOp+^conv2d_transpose_17/BiasAdd/ReadVariableOp4^conv2d_transpose_17/conv2d_transpose/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp"^dense_26/Tensordot/ReadVariableOp^embedding_10/embedding_lookup*
T0*/
_output_shapes
:џџџџџџџџџ``2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:џџџџџџџџџd:џџџџџџџџџ:::::::::::::2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2X
*conv2d_transpose_15/BiasAdd/ReadVariableOp*conv2d_transpose_15/BiasAdd/ReadVariableOp2j
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp3conv2d_transpose_15/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_16/BiasAdd/ReadVariableOp*conv2d_transpose_16/BiasAdd/ReadVariableOp2j
3conv2d_transpose_16/conv2d_transpose/ReadVariableOp3conv2d_transpose_16/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_17/BiasAdd/ReadVariableOp*conv2d_transpose_17/BiasAdd/ReadVariableOp2j
3conv2d_transpose_17/conv2d_transpose/ReadVariableOp3conv2d_transpose_17/conv2d_transpose/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2F
!dense_26/Tensordot/ReadVariableOp!dense_26/Tensordot/ReadVariableOp2>
embedding_10/embedding_lookupembedding_10/embedding_lookup:Q M
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
В
h
L__inference_leaky_re_lu_38_layer_call_and_return_conditional_losses_15572490

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
З
I
-__inference_reshape_16_layer_call_fn_15573213

inputs
identityб
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_16_layer_call_and_return_conditional_losses_155724202
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџА:T P
,
_output_shapes
:џџџџџџџџџА
 
_user_specified_nameinputs

M
1__inference_leaky_re_lu_37_layer_call_fn_15573246

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_37_layer_call_and_return_conditional_losses_155724722
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
к
х
F__inference_dense_26_layer_call_and_return_conditional_losses_15573166

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	2А*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ22
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџА2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџА2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:џџџџџџџџџА2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
гA

F__inference_model_13_layer_call_and_return_conditional_losses_15572620

inputs
inputs_1
embedding_10_15572579
dense_25_15572582
dense_25_15572584
dense_26_15572587
dense_26_15572589 
conv2d_transpose_15_15572596 
conv2d_transpose_15_15572598 
conv2d_transpose_16_15572602 
conv2d_transpose_16_15572604 
conv2d_transpose_17_15572608 
conv2d_transpose_17_15572610
conv2d_15_15572614
conv2d_15_15572616
identityЂ!conv2d_15/StatefulPartitionedCallЂ+conv2d_transpose_15/StatefulPartitionedCallЂ+conv2d_transpose_16/StatefulPartitionedCallЂ+conv2d_transpose_17/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ dense_26/StatefulPartitionedCallЂ$embedding_10/StatefulPartitionedCall
$embedding_10/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_10_15572579*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_embedding_10_layer_call_and_return_conditional_losses_155722872&
$embedding_10/StatefulPartitionedCall
 dense_25/StatefulPartitionedCallStatefulPartitionedCallinputsdense_25_15572582dense_25_15572584*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_155723092"
 dense_25/StatefulPartitionedCallЩ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall-embedding_10/StatefulPartitionedCall:output:0dense_26_15572587dense_26_15572589*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_155723552"
 dense_26/StatefulPartitionedCall
leaky_re_lu_35/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_35_layer_call_and_return_conditional_losses_155723762 
leaky_re_lu_35/PartitionedCall
reshape_15/PartitionedCallPartitionedCall'leaky_re_lu_35/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_15_layer_call_and_return_conditional_losses_155723982
reshape_15/PartitionedCall
reshape_16/PartitionedCallPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_16_layer_call_and_return_conditional_losses_155724202
reshape_16/PartitionedCallЗ
concatenate_10/PartitionedCallPartitionedCall#reshape_15/PartitionedCall:output:0#reshape_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_concatenate_10_layer_call_and_return_conditional_losses_155724352 
concatenate_10/PartitionedCall
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0conv2d_transpose_15_15572596conv2d_transpose_15_15572598*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_155721742-
+conv2d_transpose_15/StatefulPartitionedCallД
leaky_re_lu_36/PartitionedCallPartitionedCall4conv2d_transpose_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_36_layer_call_and_return_conditional_losses_155724542 
leaky_re_lu_36/PartitionedCall
+conv2d_transpose_16/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_36/PartitionedCall:output:0conv2d_transpose_16_15572602conv2d_transpose_16_15572604*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_155722182-
+conv2d_transpose_16/StatefulPartitionedCallД
leaky_re_lu_37/PartitionedCallPartitionedCall4conv2d_transpose_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_37_layer_call_and_return_conditional_losses_155724722 
leaky_re_lu_37/PartitionedCall
+conv2d_transpose_17/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_37/PartitionedCall:output:0conv2d_transpose_17_15572608conv2d_transpose_17_15572610*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_155722622-
+conv2d_transpose_17/StatefulPartitionedCallД
leaky_re_lu_38/PartitionedCallPartitionedCall4conv2d_transpose_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_38_layer_call_and_return_conditional_losses_155724902 
leaky_re_lu_38/PartitionedCallн
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_38/PartitionedCall:output:0conv2d_15_15572614conv2d_15_15572616*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_15_layer_call_and_return_conditional_losses_155725092#
!conv2d_15/StatefulPartitionedCallГ
IdentityIdentity*conv2d_15/StatefulPartitionedCall:output:0"^conv2d_15/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall,^conv2d_transpose_16/StatefulPartitionedCall,^conv2d_transpose_17/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall%^embedding_10/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:џџџџџџџџџd:џџџџџџџџџ:::::::::::::2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2Z
+conv2d_transpose_16/StatefulPartitionedCall+conv2d_transpose_16/StatefulPartitionedCall2Z
+conv2d_transpose_17/StatefulPartitionedCall+conv2d_transpose_17/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2L
$embedding_10/StatefulPartitionedCall$embedding_10/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ш

+__inference_dense_25_layer_call_fn_15573109

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_155723092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
­
M
1__inference_leaky_re_lu_35_layer_call_fn_15573136

inputs
identityЯ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_35_layer_call_and_return_conditional_losses_155723762
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ:Q M
)
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Э
h
L__inference_leaky_re_lu_35_layer_call_and_return_conditional_losses_15573131

inputs
identityV
	LeakyRelu	LeakyReluinputs*)
_output_shapes
:џџџџџџџџџ2
	LeakyRelum
IdentityIdentityLeakyRelu:activations:0*
T0*)
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_input_shapes
:џџџџџџџџџ:Q M
)
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
о
]
1__inference_concatenate_10_layer_call_fn_15573226
inputs_0
inputs_1
identityу
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_concatenate_10_layer_call_and_return_conditional_losses_155724352
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ:џџџџџџџџџ:Z V
0
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
ъ	

J__inference_embedding_10_layer_call_and_return_conditional_losses_15572287

inputs
embedding_lookup_15572281
identityЂembedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
Cast
embedding_lookupResourceGatherembedding_lookup_15572281Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/15572281*+
_output_shapes
:џџџџџџџџџ2*
dtype02
embedding_lookupя
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/15572281*+
_output_shapes
:џџџџџџџџџ22
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ22
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
єп

#__inference__wrapped_model_15572140
input_21
input_223
/model_13_embedding_10_embedding_lookup_155720124
0model_13_dense_25_matmul_readvariableop_resource5
1model_13_dense_25_biasadd_readvariableop_resource7
3model_13_dense_26_tensordot_readvariableop_resource5
1model_13_dense_26_biasadd_readvariableop_resourceI
Emodel_13_conv2d_transpose_15_conv2d_transpose_readvariableop_resource@
<model_13_conv2d_transpose_15_biasadd_readvariableop_resourceI
Emodel_13_conv2d_transpose_16_conv2d_transpose_readvariableop_resource@
<model_13_conv2d_transpose_16_biasadd_readvariableop_resourceI
Emodel_13_conv2d_transpose_17_conv2d_transpose_readvariableop_resource@
<model_13_conv2d_transpose_17_biasadd_readvariableop_resource5
1model_13_conv2d_15_conv2d_readvariableop_resource6
2model_13_conv2d_15_biasadd_readvariableop_resource
identityЂ)model_13/conv2d_15/BiasAdd/ReadVariableOpЂ(model_13/conv2d_15/Conv2D/ReadVariableOpЂ3model_13/conv2d_transpose_15/BiasAdd/ReadVariableOpЂ<model_13/conv2d_transpose_15/conv2d_transpose/ReadVariableOpЂ3model_13/conv2d_transpose_16/BiasAdd/ReadVariableOpЂ<model_13/conv2d_transpose_16/conv2d_transpose/ReadVariableOpЂ3model_13/conv2d_transpose_17/BiasAdd/ReadVariableOpЂ<model_13/conv2d_transpose_17/conv2d_transpose/ReadVariableOpЂ(model_13/dense_25/BiasAdd/ReadVariableOpЂ'model_13/dense_25/MatMul/ReadVariableOpЂ(model_13/dense_26/BiasAdd/ReadVariableOpЂ*model_13/dense_26/Tensordot/ReadVariableOpЂ&model_13/embedding_10/embedding_lookup
model_13/embedding_10/CastCastinput_22*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
model_13/embedding_10/Castя
&model_13/embedding_10/embedding_lookupResourceGather/model_13_embedding_10_embedding_lookup_15572012model_13/embedding_10/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_13/embedding_10/embedding_lookup/15572012*+
_output_shapes
:џџџџџџџџџ2*
dtype02(
&model_13/embedding_10/embedding_lookupЧ
/model_13/embedding_10/embedding_lookup/IdentityIdentity/model_13/embedding_10/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_13/embedding_10/embedding_lookup/15572012*+
_output_shapes
:џџџџџџџџџ221
/model_13/embedding_10/embedding_lookup/Identityт
1model_13/embedding_10/embedding_lookup/Identity_1Identity8model_13/embedding_10/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ223
1model_13/embedding_10/embedding_lookup/Identity_1Х
'model_13/dense_25/MatMul/ReadVariableOpReadVariableOp0model_13_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype02)
'model_13/dense_25/MatMul/ReadVariableOp­
model_13/dense_25/MatMulMatMulinput_21/model_13/dense_25/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:џџџџџџџџџ2
model_13/dense_25/MatMulФ
(model_13/dense_25/BiasAdd/ReadVariableOpReadVariableOp1model_13_dense_25_biasadd_readvariableop_resource*
_output_shapes

:*
dtype02*
(model_13/dense_25/BiasAdd/ReadVariableOpЫ
model_13/dense_25/BiasAddBiasAdd"model_13/dense_25/MatMul:product:00model_13/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:џџџџџџџџџ2
model_13/dense_25/BiasAddЭ
*model_13/dense_26/Tensordot/ReadVariableOpReadVariableOp3model_13_dense_26_tensordot_readvariableop_resource*
_output_shapes
:	2А*
dtype02,
*model_13/dense_26/Tensordot/ReadVariableOp
 model_13/dense_26/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2"
 model_13/dense_26/Tensordot/axes
 model_13/dense_26/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2"
 model_13/dense_26/Tensordot/freeА
!model_13/dense_26/Tensordot/ShapeShape:model_13/embedding_10/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2#
!model_13/dense_26/Tensordot/Shape
)model_13/dense_26/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_13/dense_26/Tensordot/GatherV2/axisЋ
$model_13/dense_26/Tensordot/GatherV2GatherV2*model_13/dense_26/Tensordot/Shape:output:0)model_13/dense_26/Tensordot/free:output:02model_13/dense_26/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$model_13/dense_26/Tensordot/GatherV2
+model_13/dense_26/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_13/dense_26/Tensordot/GatherV2_1/axisБ
&model_13/dense_26/Tensordot/GatherV2_1GatherV2*model_13/dense_26/Tensordot/Shape:output:0)model_13/dense_26/Tensordot/axes:output:04model_13/dense_26/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_13/dense_26/Tensordot/GatherV2_1
!model_13/dense_26/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model_13/dense_26/Tensordot/ConstШ
 model_13/dense_26/Tensordot/ProdProd-model_13/dense_26/Tensordot/GatherV2:output:0*model_13/dense_26/Tensordot/Const:output:0*
T0*
_output_shapes
: 2"
 model_13/dense_26/Tensordot/Prod
#model_13/dense_26/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#model_13/dense_26/Tensordot/Const_1а
"model_13/dense_26/Tensordot/Prod_1Prod/model_13/dense_26/Tensordot/GatherV2_1:output:0,model_13/dense_26/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2$
"model_13/dense_26/Tensordot/Prod_1
'model_13/dense_26/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_13/dense_26/Tensordot/concat/axis
"model_13/dense_26/Tensordot/concatConcatV2)model_13/dense_26/Tensordot/free:output:0)model_13/dense_26/Tensordot/axes:output:00model_13/dense_26/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2$
"model_13/dense_26/Tensordot/concatд
!model_13/dense_26/Tensordot/stackPack)model_13/dense_26/Tensordot/Prod:output:0+model_13/dense_26/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2#
!model_13/dense_26/Tensordot/stackњ
%model_13/dense_26/Tensordot/transpose	Transpose:model_13/embedding_10/embedding_lookup/Identity_1:output:0+model_13/dense_26/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ22'
%model_13/dense_26/Tensordot/transposeч
#model_13/dense_26/Tensordot/ReshapeReshape)model_13/dense_26/Tensordot/transpose:y:0*model_13/dense_26/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2%
#model_13/dense_26/Tensordot/Reshapeч
"model_13/dense_26/Tensordot/MatMulMatMul,model_13/dense_26/Tensordot/Reshape:output:02model_13/dense_26/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА2$
"model_13/dense_26/Tensordot/MatMul
#model_13/dense_26/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2%
#model_13/dense_26/Tensordot/Const_2
)model_13/dense_26/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_13/dense_26/Tensordot/concat_1/axis
$model_13/dense_26/Tensordot/concat_1ConcatV2-model_13/dense_26/Tensordot/GatherV2:output:0,model_13/dense_26/Tensordot/Const_2:output:02model_13/dense_26/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_13/dense_26/Tensordot/concat_1й
model_13/dense_26/TensordotReshape,model_13/dense_26/Tensordot/MatMul:product:0-model_13/dense_26/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџА2
model_13/dense_26/TensordotУ
(model_13/dense_26/BiasAdd/ReadVariableOpReadVariableOp1model_13_dense_26_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(model_13/dense_26/BiasAdd/ReadVariableOpа
model_13/dense_26/BiasAddBiasAdd$model_13/dense_26/Tensordot:output:00model_13/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџА2
model_13/dense_26/BiasAddЂ
!model_13/leaky_re_lu_35/LeakyRelu	LeakyRelu"model_13/dense_25/BiasAdd:output:0*)
_output_shapes
:џџџџџџџџџ2#
!model_13/leaky_re_lu_35/LeakyRelu
model_13/reshape_15/ShapeShape/model_13/leaky_re_lu_35/LeakyRelu:activations:0*
T0*
_output_shapes
:2
model_13/reshape_15/Shape
'model_13/reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'model_13/reshape_15/strided_slice/stack 
)model_13/reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_13/reshape_15/strided_slice/stack_1 
)model_13/reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_13/reshape_15/strided_slice/stack_2к
!model_13/reshape_15/strided_sliceStridedSlice"model_13/reshape_15/Shape:output:00model_13/reshape_15/strided_slice/stack:output:02model_13/reshape_15/strided_slice/stack_1:output:02model_13/reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!model_13/reshape_15/strided_slice
#model_13/reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#model_13/reshape_15/Reshape/shape/1
#model_13/reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#model_13/reshape_15/Reshape/shape/2
#model_13/reshape_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2%
#model_13/reshape_15/Reshape/shape/3В
!model_13/reshape_15/Reshape/shapePack*model_13/reshape_15/strided_slice:output:0,model_13/reshape_15/Reshape/shape/1:output:0,model_13/reshape_15/Reshape/shape/2:output:0,model_13/reshape_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!model_13/reshape_15/Reshape/shapeн
model_13/reshape_15/ReshapeReshape/model_13/leaky_re_lu_35/LeakyRelu:activations:0*model_13/reshape_15/Reshape/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
model_13/reshape_15/Reshape
model_13/reshape_16/ShapeShape"model_13/dense_26/BiasAdd:output:0*
T0*
_output_shapes
:2
model_13/reshape_16/Shape
'model_13/reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'model_13/reshape_16/strided_slice/stack 
)model_13/reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_13/reshape_16/strided_slice/stack_1 
)model_13/reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)model_13/reshape_16/strided_slice/stack_2к
!model_13/reshape_16/strided_sliceStridedSlice"model_13/reshape_16/Shape:output:00model_13/reshape_16/strided_slice/stack:output:02model_13/reshape_16/strided_slice/stack_1:output:02model_13/reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!model_13/reshape_16/strided_slice
#model_13/reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#model_13/reshape_16/Reshape/shape/1
#model_13/reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#model_13/reshape_16/Reshape/shape/2
#model_13/reshape_16/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#model_13/reshape_16/Reshape/shape/3В
!model_13/reshape_16/Reshape/shapePack*model_13/reshape_16/strided_slice:output:0,model_13/reshape_16/Reshape/shape/1:output:0,model_13/reshape_16/Reshape/shape/2:output:0,model_13/reshape_16/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!model_13/reshape_16/Reshape/shapeЯ
model_13/reshape_16/ReshapeReshape"model_13/dense_26/BiasAdd:output:0*model_13/reshape_16/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
model_13/reshape_16/Reshape
#model_13/concatenate_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_13/concatenate_10/concat/axis
model_13/concatenate_10/concatConcatV2$model_13/reshape_15/Reshape:output:0$model_13/reshape_16/Reshape:output:0,model_13/concatenate_10/concat/axis:output:0*
N*
T0*0
_output_shapes
:џџџџџџџџџ2 
model_13/concatenate_10/concat
"model_13/conv2d_transpose_15/ShapeShape'model_13/concatenate_10/concat:output:0*
T0*
_output_shapes
:2$
"model_13/conv2d_transpose_15/ShapeЎ
0model_13/conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_13/conv2d_transpose_15/strided_slice/stackВ
2model_13/conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_13/conv2d_transpose_15/strided_slice/stack_1В
2model_13/conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_13/conv2d_transpose_15/strided_slice/stack_2
*model_13/conv2d_transpose_15/strided_sliceStridedSlice+model_13/conv2d_transpose_15/Shape:output:09model_13/conv2d_transpose_15/strided_slice/stack:output:0;model_13/conv2d_transpose_15/strided_slice/stack_1:output:0;model_13/conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_13/conv2d_transpose_15/strided_slice
$model_13/conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_13/conv2d_transpose_15/stack/1
$model_13/conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_13/conv2d_transpose_15/stack/2
$model_13/conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value
B :2&
$model_13/conv2d_transpose_15/stack/3Р
"model_13/conv2d_transpose_15/stackPack3model_13/conv2d_transpose_15/strided_slice:output:0-model_13/conv2d_transpose_15/stack/1:output:0-model_13/conv2d_transpose_15/stack/2:output:0-model_13/conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:2$
"model_13/conv2d_transpose_15/stackВ
2model_13/conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model_13/conv2d_transpose_15/strided_slice_1/stackЖ
4model_13/conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model_13/conv2d_transpose_15/strided_slice_1/stack_1Ж
4model_13/conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model_13/conv2d_transpose_15/strided_slice_1/stack_2
,model_13/conv2d_transpose_15/strided_slice_1StridedSlice+model_13/conv2d_transpose_15/stack:output:0;model_13/conv2d_transpose_15/strided_slice_1/stack:output:0=model_13/conv2d_transpose_15/strided_slice_1/stack_1:output:0=model_13/conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model_13/conv2d_transpose_15/strided_slice_1
<model_13/conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOpEmodel_13_conv2d_transpose_15_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype02>
<model_13/conv2d_transpose_15/conv2d_transpose/ReadVariableOpє
-model_13/conv2d_transpose_15/conv2d_transposeConv2DBackpropInput+model_13/conv2d_transpose_15/stack:output:0Dmodel_13/conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0'model_13/concatenate_10/concat:output:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2/
-model_13/conv2d_transpose_15/conv2d_transposeф
3model_13/conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOp<model_13_conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_13/conv2d_transpose_15/BiasAdd/ReadVariableOp
$model_13/conv2d_transpose_15/BiasAddBiasAdd6model_13/conv2d_transpose_15/conv2d_transpose:output:0;model_13/conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2&
$model_13/conv2d_transpose_15/BiasAddД
!model_13/leaky_re_lu_36/LeakyRelu	LeakyRelu-model_13/conv2d_transpose_15/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ2#
!model_13/leaky_re_lu_36/LeakyReluЇ
"model_13/conv2d_transpose_16/ShapeShape/model_13/leaky_re_lu_36/LeakyRelu:activations:0*
T0*
_output_shapes
:2$
"model_13/conv2d_transpose_16/ShapeЎ
0model_13/conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_13/conv2d_transpose_16/strided_slice/stackВ
2model_13/conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_13/conv2d_transpose_16/strided_slice/stack_1В
2model_13/conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_13/conv2d_transpose_16/strided_slice/stack_2
*model_13/conv2d_transpose_16/strided_sliceStridedSlice+model_13/conv2d_transpose_16/Shape:output:09model_13/conv2d_transpose_16/strided_slice/stack:output:0;model_13/conv2d_transpose_16/strided_slice/stack_1:output:0;model_13/conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_13/conv2d_transpose_16/strided_slice
$model_13/conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :02&
$model_13/conv2d_transpose_16/stack/1
$model_13/conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :02&
$model_13/conv2d_transpose_16/stack/2
$model_13/conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value
B :2&
$model_13/conv2d_transpose_16/stack/3Р
"model_13/conv2d_transpose_16/stackPack3model_13/conv2d_transpose_16/strided_slice:output:0-model_13/conv2d_transpose_16/stack/1:output:0-model_13/conv2d_transpose_16/stack/2:output:0-model_13/conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:2$
"model_13/conv2d_transpose_16/stackВ
2model_13/conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model_13/conv2d_transpose_16/strided_slice_1/stackЖ
4model_13/conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model_13/conv2d_transpose_16/strided_slice_1/stack_1Ж
4model_13/conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model_13/conv2d_transpose_16/strided_slice_1/stack_2
,model_13/conv2d_transpose_16/strided_slice_1StridedSlice+model_13/conv2d_transpose_16/stack:output:0;model_13/conv2d_transpose_16/strided_slice_1/stack:output:0=model_13/conv2d_transpose_16/strided_slice_1/stack_1:output:0=model_13/conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model_13/conv2d_transpose_16/strided_slice_1
<model_13/conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOpEmodel_13_conv2d_transpose_16_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype02>
<model_13/conv2d_transpose_16/conv2d_transpose/ReadVariableOpќ
-model_13/conv2d_transpose_16/conv2d_transposeConv2DBackpropInput+model_13/conv2d_transpose_16/stack:output:0Dmodel_13/conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0/model_13/leaky_re_lu_36/LeakyRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ00*
paddingSAME*
strides
2/
-model_13/conv2d_transpose_16/conv2d_transposeф
3model_13/conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp<model_13_conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_13/conv2d_transpose_16/BiasAdd/ReadVariableOp
$model_13/conv2d_transpose_16/BiasAddBiasAdd6model_13/conv2d_transpose_16/conv2d_transpose:output:0;model_13/conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ002&
$model_13/conv2d_transpose_16/BiasAddД
!model_13/leaky_re_lu_37/LeakyRelu	LeakyRelu-model_13/conv2d_transpose_16/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ002#
!model_13/leaky_re_lu_37/LeakyReluЇ
"model_13/conv2d_transpose_17/ShapeShape/model_13/leaky_re_lu_37/LeakyRelu:activations:0*
T0*
_output_shapes
:2$
"model_13/conv2d_transpose_17/ShapeЎ
0model_13/conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_13/conv2d_transpose_17/strided_slice/stackВ
2model_13/conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_13/conv2d_transpose_17/strided_slice/stack_1В
2model_13/conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_13/conv2d_transpose_17/strided_slice/stack_2
*model_13/conv2d_transpose_17/strided_sliceStridedSlice+model_13/conv2d_transpose_17/Shape:output:09model_13/conv2d_transpose_17/strided_slice/stack:output:0;model_13/conv2d_transpose_17/strided_slice/stack_1:output:0;model_13/conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_13/conv2d_transpose_17/strided_slice
$model_13/conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`2&
$model_13/conv2d_transpose_17/stack/1
$model_13/conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value	B :`2&
$model_13/conv2d_transpose_17/stack/2
$model_13/conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value
B :2&
$model_13/conv2d_transpose_17/stack/3Р
"model_13/conv2d_transpose_17/stackPack3model_13/conv2d_transpose_17/strided_slice:output:0-model_13/conv2d_transpose_17/stack/1:output:0-model_13/conv2d_transpose_17/stack/2:output:0-model_13/conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:2$
"model_13/conv2d_transpose_17/stackВ
2model_13/conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model_13/conv2d_transpose_17/strided_slice_1/stackЖ
4model_13/conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model_13/conv2d_transpose_17/strided_slice_1/stack_1Ж
4model_13/conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model_13/conv2d_transpose_17/strided_slice_1/stack_2
,model_13/conv2d_transpose_17/strided_slice_1StridedSlice+model_13/conv2d_transpose_17/stack:output:0;model_13/conv2d_transpose_17/strided_slice_1/stack:output:0=model_13/conv2d_transpose_17/strided_slice_1/stack_1:output:0=model_13/conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model_13/conv2d_transpose_17/strided_slice_1
<model_13/conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOpEmodel_13_conv2d_transpose_17_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype02>
<model_13/conv2d_transpose_17/conv2d_transpose/ReadVariableOpќ
-model_13/conv2d_transpose_17/conv2d_transposeConv2DBackpropInput+model_13/conv2d_transpose_17/stack:output:0Dmodel_13/conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0/model_13/leaky_re_lu_37/LeakyRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ``*
paddingSAME*
strides
2/
-model_13/conv2d_transpose_17/conv2d_transposeф
3model_13/conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp<model_13_conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype025
3model_13/conv2d_transpose_17/BiasAdd/ReadVariableOp
$model_13/conv2d_transpose_17/BiasAddBiasAdd6model_13/conv2d_transpose_17/conv2d_transpose:output:0;model_13/conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ``2&
$model_13/conv2d_transpose_17/BiasAddД
!model_13/leaky_re_lu_38/LeakyRelu	LeakyRelu-model_13/conv2d_transpose_17/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ``2#
!model_13/leaky_re_lu_38/LeakyReluЯ
(model_13/conv2d_15/Conv2D/ReadVariableOpReadVariableOp1model_13_conv2d_15_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02*
(model_13/conv2d_15/Conv2D/ReadVariableOp
model_13/conv2d_15/Conv2DConv2D/model_13/leaky_re_lu_38/LeakyRelu:activations:00model_13/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ``*
paddingSAME*
strides
2
model_13/conv2d_15/Conv2DХ
)model_13/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp2model_13_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_13/conv2d_15/BiasAdd/ReadVariableOpд
model_13/conv2d_15/BiasAddBiasAdd"model_13/conv2d_15/Conv2D:output:01model_13/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ``2
model_13/conv2d_15/BiasAdd
model_13/conv2d_15/TanhTanh#model_13/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ``2
model_13/conv2d_15/Tanh
IdentityIdentitymodel_13/conv2d_15/Tanh:y:0*^model_13/conv2d_15/BiasAdd/ReadVariableOp)^model_13/conv2d_15/Conv2D/ReadVariableOp4^model_13/conv2d_transpose_15/BiasAdd/ReadVariableOp=^model_13/conv2d_transpose_15/conv2d_transpose/ReadVariableOp4^model_13/conv2d_transpose_16/BiasAdd/ReadVariableOp=^model_13/conv2d_transpose_16/conv2d_transpose/ReadVariableOp4^model_13/conv2d_transpose_17/BiasAdd/ReadVariableOp=^model_13/conv2d_transpose_17/conv2d_transpose/ReadVariableOp)^model_13/dense_25/BiasAdd/ReadVariableOp(^model_13/dense_25/MatMul/ReadVariableOp)^model_13/dense_26/BiasAdd/ReadVariableOp+^model_13/dense_26/Tensordot/ReadVariableOp'^model_13/embedding_10/embedding_lookup*
T0*/
_output_shapes
:џџџџџџџџџ``2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:џџџџџџџџџd:џџџџџџџџџ:::::::::::::2V
)model_13/conv2d_15/BiasAdd/ReadVariableOp)model_13/conv2d_15/BiasAdd/ReadVariableOp2T
(model_13/conv2d_15/Conv2D/ReadVariableOp(model_13/conv2d_15/Conv2D/ReadVariableOp2j
3model_13/conv2d_transpose_15/BiasAdd/ReadVariableOp3model_13/conv2d_transpose_15/BiasAdd/ReadVariableOp2|
<model_13/conv2d_transpose_15/conv2d_transpose/ReadVariableOp<model_13/conv2d_transpose_15/conv2d_transpose/ReadVariableOp2j
3model_13/conv2d_transpose_16/BiasAdd/ReadVariableOp3model_13/conv2d_transpose_16/BiasAdd/ReadVariableOp2|
<model_13/conv2d_transpose_16/conv2d_transpose/ReadVariableOp<model_13/conv2d_transpose_16/conv2d_transpose/ReadVariableOp2j
3model_13/conv2d_transpose_17/BiasAdd/ReadVariableOp3model_13/conv2d_transpose_17/BiasAdd/ReadVariableOp2|
<model_13/conv2d_transpose_17/conv2d_transpose/ReadVariableOp<model_13/conv2d_transpose_17/conv2d_transpose/ReadVariableOp2T
(model_13/dense_25/BiasAdd/ReadVariableOp(model_13/dense_25/BiasAdd/ReadVariableOp2R
'model_13/dense_25/MatMul/ReadVariableOp'model_13/dense_25/MatMul/ReadVariableOp2T
(model_13/dense_26/BiasAdd/ReadVariableOp(model_13/dense_26/BiasAdd/ReadVariableOp2X
*model_13/dense_26/Tensordot/ReadVariableOp*model_13/dense_26/Tensordot/ReadVariableOp2P
&model_13/embedding_10/embedding_lookup&model_13/embedding_10/embedding_lookup:Q M
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
input_21:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_22
В
h
L__inference_leaky_re_lu_36_layer_call_and_return_conditional_losses_15572454

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Е

В
&__inference_signature_wrapper_15572760
input_21
input_22
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identityЂStatefulPartitionedCallќ
StatefulPartitionedCallStatefulPartitionedCallinput_21input_22unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ``*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_155721402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ``2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:џџџџџџџџџd:џџџџџџџџџ:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
input_21:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_22

З
+__inference_model_13_layer_call_fn_15573058
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identityЂStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_model_13_layer_call_and_return_conditional_losses_155726202
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:џџџџџџџџџd:џџџџџџџџџ:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
лA

F__inference_model_13_layer_call_and_return_conditional_losses_15572571
input_21
input_22
embedding_10_15572530
dense_25_15572533
dense_25_15572535
dense_26_15572538
dense_26_15572540 
conv2d_transpose_15_15572547 
conv2d_transpose_15_15572549 
conv2d_transpose_16_15572553 
conv2d_transpose_16_15572555 
conv2d_transpose_17_15572559 
conv2d_transpose_17_15572561
conv2d_15_15572565
conv2d_15_15572567
identityЂ!conv2d_15/StatefulPartitionedCallЂ+conv2d_transpose_15/StatefulPartitionedCallЂ+conv2d_transpose_16/StatefulPartitionedCallЂ+conv2d_transpose_17/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ dense_26/StatefulPartitionedCallЂ$embedding_10/StatefulPartitionedCall
$embedding_10/StatefulPartitionedCallStatefulPartitionedCallinput_22embedding_10_15572530*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_embedding_10_layer_call_and_return_conditional_losses_155722872&
$embedding_10/StatefulPartitionedCallЁ
 dense_25/StatefulPartitionedCallStatefulPartitionedCallinput_21dense_25_15572533dense_25_15572535*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_155723092"
 dense_25/StatefulPartitionedCallЩ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall-embedding_10/StatefulPartitionedCall:output:0dense_26_15572538dense_26_15572540*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_155723552"
 dense_26/StatefulPartitionedCall
leaky_re_lu_35/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_35_layer_call_and_return_conditional_losses_155723762 
leaky_re_lu_35/PartitionedCall
reshape_15/PartitionedCallPartitionedCall'leaky_re_lu_35/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_15_layer_call_and_return_conditional_losses_155723982
reshape_15/PartitionedCall
reshape_16/PartitionedCallPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_16_layer_call_and_return_conditional_losses_155724202
reshape_16/PartitionedCallЗ
concatenate_10/PartitionedCallPartitionedCall#reshape_15/PartitionedCall:output:0#reshape_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_concatenate_10_layer_call_and_return_conditional_losses_155724352 
concatenate_10/PartitionedCall
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0conv2d_transpose_15_15572547conv2d_transpose_15_15572549*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_155721742-
+conv2d_transpose_15/StatefulPartitionedCallД
leaky_re_lu_36/PartitionedCallPartitionedCall4conv2d_transpose_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_36_layer_call_and_return_conditional_losses_155724542 
leaky_re_lu_36/PartitionedCall
+conv2d_transpose_16/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_36/PartitionedCall:output:0conv2d_transpose_16_15572553conv2d_transpose_16_15572555*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_155722182-
+conv2d_transpose_16/StatefulPartitionedCallД
leaky_re_lu_37/PartitionedCallPartitionedCall4conv2d_transpose_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_37_layer_call_and_return_conditional_losses_155724722 
leaky_re_lu_37/PartitionedCall
+conv2d_transpose_17/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_37/PartitionedCall:output:0conv2d_transpose_17_15572559conv2d_transpose_17_15572561*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_155722622-
+conv2d_transpose_17/StatefulPartitionedCallД
leaky_re_lu_38/PartitionedCallPartitionedCall4conv2d_transpose_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_38_layer_call_and_return_conditional_losses_155724902 
leaky_re_lu_38/PartitionedCallн
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_38/PartitionedCall:output:0conv2d_15_15572565conv2d_15_15572567*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_15_layer_call_and_return_conditional_losses_155725092#
!conv2d_15/StatefulPartitionedCallГ
IdentityIdentity*conv2d_15/StatefulPartitionedCall:output:0"^conv2d_15/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall,^conv2d_transpose_16/StatefulPartitionedCall,^conv2d_transpose_17/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall%^embedding_10/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:џџџџџџџџџd:џџџџџџџџџ:::::::::::::2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2Z
+conv2d_transpose_16/StatefulPartitionedCall+conv2d_transpose_16/StatefulPartitionedCall2Z
+conv2d_transpose_17/StatefulPartitionedCall+conv2d_transpose_17/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2L
$embedding_10/StatefulPartitionedCall$embedding_10/StatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
input_21:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_22
э
v
L__inference_concatenate_10_layer_call_and_return_conditional_losses_15572435

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:џџџџџџџџџ2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:џџџџџџџџџ:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
лA

F__inference_model_13_layer_call_and_return_conditional_losses_15572526
input_21
input_22
embedding_10_15572296
dense_25_15572320
dense_25_15572322
dense_26_15572366
dense_26_15572368 
conv2d_transpose_15_15572444 
conv2d_transpose_15_15572446 
conv2d_transpose_16_15572462 
conv2d_transpose_16_15572464 
conv2d_transpose_17_15572480 
conv2d_transpose_17_15572482
conv2d_15_15572520
conv2d_15_15572522
identityЂ!conv2d_15/StatefulPartitionedCallЂ+conv2d_transpose_15/StatefulPartitionedCallЂ+conv2d_transpose_16/StatefulPartitionedCallЂ+conv2d_transpose_17/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ dense_26/StatefulPartitionedCallЂ$embedding_10/StatefulPartitionedCall
$embedding_10/StatefulPartitionedCallStatefulPartitionedCallinput_22embedding_10_15572296*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_embedding_10_layer_call_and_return_conditional_losses_155722872&
$embedding_10/StatefulPartitionedCallЁ
 dense_25/StatefulPartitionedCallStatefulPartitionedCallinput_21dense_25_15572320dense_25_15572322*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_155723092"
 dense_25/StatefulPartitionedCallЩ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall-embedding_10/StatefulPartitionedCall:output:0dense_26_15572366dense_26_15572368*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_155723552"
 dense_26/StatefulPartitionedCall
leaky_re_lu_35/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_35_layer_call_and_return_conditional_losses_155723762 
leaky_re_lu_35/PartitionedCall
reshape_15/PartitionedCallPartitionedCall'leaky_re_lu_35/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_15_layer_call_and_return_conditional_losses_155723982
reshape_15/PartitionedCall
reshape_16/PartitionedCallPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_16_layer_call_and_return_conditional_losses_155724202
reshape_16/PartitionedCallЗ
concatenate_10/PartitionedCallPartitionedCall#reshape_15/PartitionedCall:output:0#reshape_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_concatenate_10_layer_call_and_return_conditional_losses_155724352 
concatenate_10/PartitionedCall
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0conv2d_transpose_15_15572444conv2d_transpose_15_15572446*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_155721742-
+conv2d_transpose_15/StatefulPartitionedCallД
leaky_re_lu_36/PartitionedCallPartitionedCall4conv2d_transpose_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_36_layer_call_and_return_conditional_losses_155724542 
leaky_re_lu_36/PartitionedCall
+conv2d_transpose_16/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_36/PartitionedCall:output:0conv2d_transpose_16_15572462conv2d_transpose_16_15572464*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_155722182-
+conv2d_transpose_16/StatefulPartitionedCallД
leaky_re_lu_37/PartitionedCallPartitionedCall4conv2d_transpose_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_37_layer_call_and_return_conditional_losses_155724722 
leaky_re_lu_37/PartitionedCall
+conv2d_transpose_17/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_37/PartitionedCall:output:0conv2d_transpose_17_15572480conv2d_transpose_17_15572482*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_155722622-
+conv2d_transpose_17/StatefulPartitionedCallД
leaky_re_lu_38/PartitionedCallPartitionedCall4conv2d_transpose_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_38_layer_call_and_return_conditional_losses_155724902 
leaky_re_lu_38/PartitionedCallн
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_38/PartitionedCall:output:0conv2d_15_15572520conv2d_15_15572522*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_15_layer_call_and_return_conditional_losses_155725092#
!conv2d_15/StatefulPartitionedCallГ
IdentityIdentity*conv2d_15/StatefulPartitionedCall:output:0"^conv2d_15/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall,^conv2d_transpose_16/StatefulPartitionedCall,^conv2d_transpose_17/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall%^embedding_10/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:џџџџџџџџџd:џџџџџџџџџ:::::::::::::2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2Z
+conv2d_transpose_16/StatefulPartitionedCall+conv2d_transpose_16/StatefulPartitionedCall2Z
+conv2d_transpose_17/StatefulPartitionedCall+conv2d_transpose_17/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2L
$embedding_10/StatefulPartitionedCall$embedding_10/StatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
input_21:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_22
ј#
ў
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_15572174

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :2	
stack/3
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ь
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3Е
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpё
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddК
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
бС
П

F__inference_model_13_layer_call_and_return_conditional_losses_15573026
inputs_0
inputs_1*
&embedding_10_embedding_lookup_15572898+
'dense_25_matmul_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource.
*dense_26_tensordot_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource@
<conv2d_transpose_15_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_15_biasadd_readvariableop_resource@
<conv2d_transpose_16_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_16_biasadd_readvariableop_resource@
<conv2d_transpose_17_conv2d_transpose_readvariableop_resource7
3conv2d_transpose_17_biasadd_readvariableop_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource
identityЂ conv2d_15/BiasAdd/ReadVariableOpЂconv2d_15/Conv2D/ReadVariableOpЂ*conv2d_transpose_15/BiasAdd/ReadVariableOpЂ3conv2d_transpose_15/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_16/BiasAdd/ReadVariableOpЂ3conv2d_transpose_16/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_17/BiasAdd/ReadVariableOpЂ3conv2d_transpose_17/conv2d_transpose/ReadVariableOpЂdense_25/BiasAdd/ReadVariableOpЂdense_25/MatMul/ReadVariableOpЂdense_26/BiasAdd/ReadVariableOpЂ!dense_26/Tensordot/ReadVariableOpЂembedding_10/embedding_lookupy
embedding_10/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
embedding_10/CastТ
embedding_10/embedding_lookupResourceGather&embedding_10_embedding_lookup_15572898embedding_10/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_10/embedding_lookup/15572898*+
_output_shapes
:џџџџџџџџџ2*
dtype02
embedding_10/embedding_lookupЃ
&embedding_10/embedding_lookup/IdentityIdentity&embedding_10/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_10/embedding_lookup/15572898*+
_output_shapes
:џџџџџџџџџ22(
&embedding_10/embedding_lookup/IdentityЧ
(embedding_10/embedding_lookup/Identity_1Identity/embedding_10/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ22*
(embedding_10/embedding_lookup/Identity_1Њ
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype02 
dense_25/MatMul/ReadVariableOp
dense_25/MatMulMatMulinputs_0&dense_25/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:џџџџџџџџџ2
dense_25/MatMulЉ
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_25/BiasAdd/ReadVariableOpЇ
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:џџџџџџџџџ2
dense_25/BiasAddВ
!dense_26/Tensordot/ReadVariableOpReadVariableOp*dense_26_tensordot_readvariableop_resource*
_output_shapes
:	2А*
dtype02#
!dense_26/Tensordot/ReadVariableOp|
dense_26/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_26/Tensordot/axes
dense_26/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_26/Tensordot/free
dense_26/Tensordot/ShapeShape1embedding_10/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense_26/Tensordot/Shape
 dense_26/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_26/Tensordot/GatherV2/axisў
dense_26/Tensordot/GatherV2GatherV2!dense_26/Tensordot/Shape:output:0 dense_26/Tensordot/free:output:0)dense_26/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_26/Tensordot/GatherV2
"dense_26/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_26/Tensordot/GatherV2_1/axis
dense_26/Tensordot/GatherV2_1GatherV2!dense_26/Tensordot/Shape:output:0 dense_26/Tensordot/axes:output:0+dense_26/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_26/Tensordot/GatherV2_1~
dense_26/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_26/Tensordot/ConstЄ
dense_26/Tensordot/ProdProd$dense_26/Tensordot/GatherV2:output:0!dense_26/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_26/Tensordot/Prod
dense_26/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_26/Tensordot/Const_1Ќ
dense_26/Tensordot/Prod_1Prod&dense_26/Tensordot/GatherV2_1:output:0#dense_26/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_26/Tensordot/Prod_1
dense_26/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_26/Tensordot/concat/axisн
dense_26/Tensordot/concatConcatV2 dense_26/Tensordot/free:output:0 dense_26/Tensordot/axes:output:0'dense_26/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_26/Tensordot/concatА
dense_26/Tensordot/stackPack dense_26/Tensordot/Prod:output:0"dense_26/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_26/Tensordot/stackж
dense_26/Tensordot/transpose	Transpose1embedding_10/embedding_lookup/Identity_1:output:0"dense_26/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ22
dense_26/Tensordot/transposeУ
dense_26/Tensordot/ReshapeReshape dense_26/Tensordot/transpose:y:0!dense_26/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_26/Tensordot/ReshapeУ
dense_26/Tensordot/MatMulMatMul#dense_26/Tensordot/Reshape:output:0)dense_26/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџА2
dense_26/Tensordot/MatMul
dense_26/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
dense_26/Tensordot/Const_2
 dense_26/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_26/Tensordot/concat_1/axisъ
dense_26/Tensordot/concat_1ConcatV2$dense_26/Tensordot/GatherV2:output:0#dense_26/Tensordot/Const_2:output:0)dense_26/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_26/Tensordot/concat_1Е
dense_26/TensordotReshape#dense_26/Tensordot/MatMul:product:0$dense_26/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџА2
dense_26/TensordotЈ
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_26/BiasAdd/ReadVariableOpЌ
dense_26/BiasAddBiasAdddense_26/Tensordot:output:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџА2
dense_26/BiasAdd
leaky_re_lu_35/LeakyRelu	LeakyReludense_25/BiasAdd:output:0*)
_output_shapes
:џџџџџџџџџ2
leaky_re_lu_35/LeakyReluz
reshape_15/ShapeShape&leaky_re_lu_35/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_15/Shape
reshape_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_15/strided_slice/stack
 reshape_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_1
 reshape_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_15/strided_slice/stack_2Є
reshape_15/strided_sliceStridedSlicereshape_15/Shape:output:0'reshape_15/strided_slice/stack:output:0)reshape_15/strided_slice/stack_1:output:0)reshape_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_15/strided_slicez
reshape_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_15/Reshape/shape/1z
reshape_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_15/Reshape/shape/2{
reshape_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
reshape_15/Reshape/shape/3ќ
reshape_15/Reshape/shapePack!reshape_15/strided_slice:output:0#reshape_15/Reshape/shape/1:output:0#reshape_15/Reshape/shape/2:output:0#reshape_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_15/Reshape/shapeЙ
reshape_15/ReshapeReshape&leaky_re_lu_35/LeakyRelu:activations:0!reshape_15/Reshape/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
reshape_15/Reshapem
reshape_16/ShapeShapedense_26/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_16/Shape
reshape_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_16/strided_slice/stack
 reshape_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_1
 reshape_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_16/strided_slice/stack_2Є
reshape_16/strided_sliceStridedSlicereshape_16/Shape:output:0'reshape_16/strided_slice/stack:output:0)reshape_16/strided_slice/stack_1:output:0)reshape_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_16/strided_slicez
reshape_16/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/1z
reshape_16/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/2z
reshape_16/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_16/Reshape/shape/3ќ
reshape_16/Reshape/shapePack!reshape_16/strided_slice:output:0#reshape_16/Reshape/shape/1:output:0#reshape_16/Reshape/shape/2:output:0#reshape_16/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_16/Reshape/shapeЋ
reshape_16/ReshapeReshapedense_26/BiasAdd:output:0!reshape_16/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
reshape_16/Reshapez
concatenate_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_10/concat/axisн
concatenate_10/concatConcatV2reshape_15/Reshape:output:0reshape_16/Reshape:output:0#concatenate_10/concat/axis:output:0*
N*
T0*0
_output_shapes
:џџџџџџџџџ2
concatenate_10/concat
conv2d_transpose_15/ShapeShapeconcatenate_10/concat:output:0*
T0*
_output_shapes
:2
conv2d_transpose_15/Shape
'conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_15/strided_slice/stack 
)conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_15/strided_slice/stack_1 
)conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_15/strided_slice/stack_2к
!conv2d_transpose_15/strided_sliceStridedSlice"conv2d_transpose_15/Shape:output:00conv2d_transpose_15/strided_slice/stack:output:02conv2d_transpose_15/strided_slice/stack_1:output:02conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_15/strided_slice|
conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_15/stack/1|
conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_15/stack/2}
conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value
B :2
conv2d_transpose_15/stack/3
conv2d_transpose_15/stackPack*conv2d_transpose_15/strided_slice:output:0$conv2d_transpose_15/stack/1:output:0$conv2d_transpose_15/stack/2:output:0$conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_15/stack 
)conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_15/strided_slice_1/stackЄ
+conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_15/strided_slice_1/stack_1Є
+conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_15/strided_slice_1/stack_2ф
#conv2d_transpose_15/strided_slice_1StridedSlice"conv2d_transpose_15/stack:output:02conv2d_transpose_15/strided_slice_1/stack:output:04conv2d_transpose_15/strided_slice_1/stack_1:output:04conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_15/strided_slice_1ё
3conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_15_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype025
3conv2d_transpose_15/conv2d_transpose/ReadVariableOpЧ
$conv2d_transpose_15/conv2d_transposeConv2DBackpropInput"conv2d_transpose_15/stack:output:0;conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0concatenate_10/concat:output:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2&
$conv2d_transpose_15/conv2d_transposeЩ
*conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*conv2d_transpose_15/BiasAdd/ReadVariableOpу
conv2d_transpose_15/BiasAddBiasAdd-conv2d_transpose_15/conv2d_transpose:output:02conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_transpose_15/BiasAdd
leaky_re_lu_36/LeakyRelu	LeakyRelu$conv2d_transpose_15/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ2
leaky_re_lu_36/LeakyRelu
conv2d_transpose_16/ShapeShape&leaky_re_lu_36/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_16/Shape
'conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_16/strided_slice/stack 
)conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_16/strided_slice/stack_1 
)conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_16/strided_slice/stack_2к
!conv2d_transpose_16/strided_sliceStridedSlice"conv2d_transpose_16/Shape:output:00conv2d_transpose_16/strided_slice/stack:output:02conv2d_transpose_16/strided_slice/stack_1:output:02conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_16/strided_slice|
conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :02
conv2d_transpose_16/stack/1|
conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :02
conv2d_transpose_16/stack/2}
conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value
B :2
conv2d_transpose_16/stack/3
conv2d_transpose_16/stackPack*conv2d_transpose_16/strided_slice:output:0$conv2d_transpose_16/stack/1:output:0$conv2d_transpose_16/stack/2:output:0$conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_16/stack 
)conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_16/strided_slice_1/stackЄ
+conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_16/strided_slice_1/stack_1Є
+conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_16/strided_slice_1/stack_2ф
#conv2d_transpose_16/strided_slice_1StridedSlice"conv2d_transpose_16/stack:output:02conv2d_transpose_16/strided_slice_1/stack:output:04conv2d_transpose_16/strided_slice_1/stack_1:output:04conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_16/strided_slice_1ё
3conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_16_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype025
3conv2d_transpose_16/conv2d_transpose/ReadVariableOpЯ
$conv2d_transpose_16/conv2d_transposeConv2DBackpropInput"conv2d_transpose_16/stack:output:0;conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_36/LeakyRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ00*
paddingSAME*
strides
2&
$conv2d_transpose_16/conv2d_transposeЩ
*conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*conv2d_transpose_16/BiasAdd/ReadVariableOpу
conv2d_transpose_16/BiasAddBiasAdd-conv2d_transpose_16/conv2d_transpose:output:02conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ002
conv2d_transpose_16/BiasAdd
leaky_re_lu_37/LeakyRelu	LeakyRelu$conv2d_transpose_16/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ002
leaky_re_lu_37/LeakyRelu
conv2d_transpose_17/ShapeShape&leaky_re_lu_37/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_17/Shape
'conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_17/strided_slice/stack 
)conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_17/strided_slice/stack_1 
)conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_17/strided_slice/stack_2к
!conv2d_transpose_17/strided_sliceStridedSlice"conv2d_transpose_17/Shape:output:00conv2d_transpose_17/strided_slice/stack:output:02conv2d_transpose_17/strided_slice/stack_1:output:02conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_17/strided_slice|
conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`2
conv2d_transpose_17/stack/1|
conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value	B :`2
conv2d_transpose_17/stack/2}
conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value
B :2
conv2d_transpose_17/stack/3
conv2d_transpose_17/stackPack*conv2d_transpose_17/strided_slice:output:0$conv2d_transpose_17/stack/1:output:0$conv2d_transpose_17/stack/2:output:0$conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_17/stack 
)conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_17/strided_slice_1/stackЄ
+conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_17/strided_slice_1/stack_1Є
+conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_17/strided_slice_1/stack_2ф
#conv2d_transpose_17/strided_slice_1StridedSlice"conv2d_transpose_17/stack:output:02conv2d_transpose_17/strided_slice_1/stack:output:04conv2d_transpose_17/strided_slice_1/stack_1:output:04conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_17/strided_slice_1ё
3conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_17_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype025
3conv2d_transpose_17/conv2d_transpose/ReadVariableOpЯ
$conv2d_transpose_17/conv2d_transposeConv2DBackpropInput"conv2d_transpose_17/stack:output:0;conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_37/LeakyRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ``*
paddingSAME*
strides
2&
$conv2d_transpose_17/conv2d_transposeЩ
*conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*conv2d_transpose_17/BiasAdd/ReadVariableOpу
conv2d_transpose_17/BiasAddBiasAdd-conv2d_transpose_17/conv2d_transpose:output:02conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ``2
conv2d_transpose_17/BiasAdd
leaky_re_lu_38/LeakyRelu	LeakyRelu$conv2d_transpose_17/BiasAdd:output:0*0
_output_shapes
:џџџџџџџџџ``2
leaky_re_lu_38/LeakyReluД
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02!
conv2d_15/Conv2D/ReadVariableOpс
conv2d_15/Conv2DConv2D&leaky_re_lu_38/LeakyRelu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ``*
paddingSAME*
strides
2
conv2d_15/Conv2DЊ
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOpА
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ``2
conv2d_15/BiasAdd~
conv2d_15/TanhTanhconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ``2
conv2d_15/Tanh
IdentityIdentityconv2d_15/Tanh:y:0!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp+^conv2d_transpose_15/BiasAdd/ReadVariableOp4^conv2d_transpose_15/conv2d_transpose/ReadVariableOp+^conv2d_transpose_16/BiasAdd/ReadVariableOp4^conv2d_transpose_16/conv2d_transpose/ReadVariableOp+^conv2d_transpose_17/BiasAdd/ReadVariableOp4^conv2d_transpose_17/conv2d_transpose/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp"^dense_26/Tensordot/ReadVariableOp^embedding_10/embedding_lookup*
T0*/
_output_shapes
:џџџџџџџџџ``2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:џџџџџџџџџd:џџџџџџџџџ:::::::::::::2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2X
*conv2d_transpose_15/BiasAdd/ReadVariableOp*conv2d_transpose_15/BiasAdd/ReadVariableOp2j
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp3conv2d_transpose_15/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_16/BiasAdd/ReadVariableOp*conv2d_transpose_16/BiasAdd/ReadVariableOp2j
3conv2d_transpose_16/conv2d_transpose/ReadVariableOp3conv2d_transpose_16/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_17/BiasAdd/ReadVariableOp*conv2d_transpose_17/BiasAdd/ReadVariableOp2j
3conv2d_transpose_17/conv2d_transpose/ReadVariableOp3conv2d_transpose_17/conv2d_transpose/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2F
!dense_26/Tensordot/ReadVariableOp!dense_26/Tensordot/ReadVariableOp2>
embedding_10/embedding_lookupembedding_10/embedding_lookup:Q M
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1

З
+__inference_model_13_layer_call_fn_15572726
input_21
input_22
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identityЂStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinput_21input_22unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_model_13_layer_call_and_return_conditional_losses_155726972
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:џџџџџџџџџd:џџџџџџџџџ:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
input_21:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_22
ј#
ў
Q__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_15572262

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ь
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :2	
stack/3
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2ь
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3Е
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpё
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddК
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
В
h
L__inference_leaky_re_lu_36_layer_call_and_return_conditional_losses_15573231

inputs
identityo
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
	LeakyRelu
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ъ	

J__inference_embedding_10_layer_call_and_return_conditional_losses_15573119

inputs
embedding_lookup_15573113
identityЂembedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
Cast
embedding_lookupResourceGatherembedding_lookup_15573113Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/15573113*+
_output_shapes
:џџџџџџџџџ2*
dtype02
embedding_lookupя
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/15573113*+
_output_shapes
:џџџџџџџџџ22
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ22
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѓ
d
H__inference_reshape_16_layer_call_and_return_conditional_losses_15573208

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3К
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџА:T P
,
_output_shapes
:џџџџџџџџџА
 
_user_specified_nameinputs
ц

6__inference_conv2d_transpose_16_layer_call_fn_15572228

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_155722182
StatefulPartitionedCallЉ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

M
1__inference_leaky_re_lu_36_layer_call_fn_15573236

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_leaky_re_lu_36_layer_call_and_return_conditional_losses_155724542
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ѕ
serving_defaultс
=
input_211
serving_default_input_21:0џџџџџџџџџd
=
input_221
serving_default_input_22:0џџџџџџџџџE
	conv2d_158
StatefulPartitionedCall:0џџџџџџџџџ``tensorflow/serving/predict:бЅ
лy
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer_with_weights-5
layer-13
layer-14
layer_with_weights-6
layer-15
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
Ї__call__
Ј_default_save_signature
+Љ&call_and_return_all_conditional_losses"u
_tf_keras_networkхt{"class_name": "Functional", "name": "model_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_13", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_21"}, "name": "input_21", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_22"}, "name": "input_22", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "units": 18432, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_25", "inbound_nodes": [[["input_21", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 2, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_10", "inbound_nodes": [[["input_22", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_35", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_35", "inbound_nodes": [[["dense_25", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 432, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_26", "inbound_nodes": [[["embedding_10", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_15", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [12, 12, 128]}}, "name": "reshape_15", "inbound_nodes": [[["leaky_re_lu_35", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [12, 12, 3]}}, "name": "reshape_16", "inbound_nodes": [[["dense_26", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_10", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_10", "inbound_nodes": [[["reshape_15", 0, 0, {}], ["reshape_16", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_15", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_15", "inbound_nodes": [[["concatenate_10", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_36", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_36", "inbound_nodes": [[["conv2d_transpose_15", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_16", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_16", "inbound_nodes": [[["leaky_re_lu_36", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_37", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_37", "inbound_nodes": [[["conv2d_transpose_16", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_17", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_17", "inbound_nodes": [[["leaky_re_lu_37", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_38", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_38", "inbound_nodes": [[["conv2d_transpose_17", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["leaky_re_lu_38", 0, 0, {}]]]}], "input_layers": [["input_21", 0, 0], ["input_22", 0, 0]], "output_layers": [["conv2d_15", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 100]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 100]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_13", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_21"}, "name": "input_21", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_22"}, "name": "input_22", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "units": 18432, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_25", "inbound_nodes": [[["input_21", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 2, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_10", "inbound_nodes": [[["input_22", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_35", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_35", "inbound_nodes": [[["dense_25", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 432, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_26", "inbound_nodes": [[["embedding_10", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_15", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [12, 12, 128]}}, "name": "reshape_15", "inbound_nodes": [[["leaky_re_lu_35", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [12, 12, 3]}}, "name": "reshape_16", "inbound_nodes": [[["dense_26", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_10", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_10", "inbound_nodes": [[["reshape_15", 0, 0, {}], ["reshape_16", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_15", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_15", "inbound_nodes": [[["concatenate_10", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_36", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_36", "inbound_nodes": [[["conv2d_transpose_15", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_16", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_16", "inbound_nodes": [[["leaky_re_lu_36", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_37", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_37", "inbound_nodes": [[["conv2d_transpose_16", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_17", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_17", "inbound_nodes": [[["leaky_re_lu_37", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_38", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_38", "inbound_nodes": [[["conv2d_transpose_17", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["leaky_re_lu_38", 0, 0, {}]]]}], "input_layers": [["input_21", 0, 0], ["input_22", 0, 0]], "output_layers": [["conv2d_15", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0002, "decay": 0.0, "beta_1": 0.5, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
я"ь
_tf_keras_input_layerЬ{"class_name": "InputLayer", "name": "input_21", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_21"}}
ы"ш
_tf_keras_input_layerШ{"class_name": "InputLayer", "name": "input_22", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_22"}}
	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
Њ__call__
+Ћ&call_and_return_all_conditional_losses"х
_tf_keras_layerЫ{"class_name": "Dense", "name": "dense_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_25", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "units": 18432, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
Ў

embeddings
trainable_variables
regularization_losses
 	variables
!	keras_api
Ќ__call__
+­&call_and_return_all_conditional_losses"
_tf_keras_layerѓ{"class_name": "Embedding", "name": "embedding_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 2, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
т
"trainable_variables
#regularization_losses
$	variables
%	keras_api
Ў__call__
+Џ&call_and_return_all_conditional_losses"б
_tf_keras_layerЗ{"class_name": "LeakyReLU", "name": "leaky_re_lu_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_35", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
њ

&kernel
'bias
(trainable_variables
)regularization_losses
*	variables
+	keras_api
А__call__
+Б&call_and_return_all_conditional_losses"г
_tf_keras_layerЙ{"class_name": "Dense", "name": "dense_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 432, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 50]}}
џ
,trainable_variables
-regularization_losses
.	variables
/	keras_api
В__call__
+Г&call_and_return_all_conditional_losses"ю
_tf_keras_layerд{"class_name": "Reshape", "name": "reshape_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_15", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [12, 12, 128]}}}
§
0trainable_variables
1regularization_losses
2	variables
3	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"ь
_tf_keras_layerв{"class_name": "Reshape", "name": "reshape_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_16", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [12, 12, 3]}}}
с
4trainable_variables
5regularization_losses
6	variables
7	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"class_name": "Concatenate", "name": "concatenate_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_10", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 12, 12, 128]}, {"class_name": "TensorShape", "items": [null, 12, 12, 3]}]}
А


8kernel
9bias
:trainable_variables
;regularization_losses
<	variables
=	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"	
_tf_keras_layerя{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_15", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 131}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 131]}}
т
>trainable_variables
?regularization_losses
@	variables
A	keras_api
К__call__
+Л&call_and_return_all_conditional_losses"б
_tf_keras_layerЗ{"class_name": "LeakyReLU", "name": "leaky_re_lu_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_36", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
А


Bkernel
Cbias
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
М__call__
+Н&call_and_return_all_conditional_losses"	
_tf_keras_layerя{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_16", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 128]}}
т
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
О__call__
+П&call_and_return_all_conditional_losses"б
_tf_keras_layerЗ{"class_name": "LeakyReLU", "name": "leaky_re_lu_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_37", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
А


Lkernel
Mbias
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"	
_tf_keras_layerя{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_17", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 128]}}
т
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
Т__call__
+У&call_and_return_all_conditional_losses"б
_tf_keras_layerЗ{"class_name": "LeakyReLU", "name": "leaky_re_lu_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_38", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
ї	

Vkernel
Wbias
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"class_name": "Conv2D", "name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 96, 128]}}
"
	optimizer
~
0
1
2
&3
'4
85
96
B7
C8
L9
M10
V11
W12"
trackable_list_wrapper
 "
trackable_list_wrapper
~
0
1
2
&3
'4
85
96
B7
C8
L9
M10
V11
W12"
trackable_list_wrapper
Ю
\layer_metrics

]layers
^metrics
_non_trainable_variables
`layer_regularization_losses
trainable_variables
regularization_losses
	variables
Ї__call__
Ј_default_save_signature
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
-
Цserving_default"
signature_map
#:!
d2dense_25/kernel
:2dense_25/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А
alayer_metrics

blayers
cmetrics
dnon_trainable_variables
elayer_regularization_losses
trainable_variables
regularization_losses
	variables
Њ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
):'22embedding_10/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
А
flayer_metrics

glayers
hmetrics
inon_trainable_variables
jlayer_regularization_losses
trainable_variables
regularization_losses
 	variables
Ќ__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
klayer_metrics

llayers
mmetrics
nnon_trainable_variables
olayer_regularization_losses
"trainable_variables
#regularization_losses
$	variables
Ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
": 	2А2dense_26/kernel
:А2dense_26/bias
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
А
player_metrics

qlayers
rmetrics
snon_trainable_variables
tlayer_regularization_losses
(trainable_variables
)regularization_losses
*	variables
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
ulayer_metrics

vlayers
wmetrics
xnon_trainable_variables
ylayer_regularization_losses
,trainable_variables
-regularization_losses
.	variables
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
zlayer_metrics

{layers
|metrics
}non_trainable_variables
~layer_regularization_losses
0trainable_variables
1regularization_losses
2	variables
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Д
layer_metrics
layers
metrics
non_trainable_variables
 layer_regularization_losses
4trainable_variables
5regularization_losses
6	variables
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
6:42conv2d_transpose_15/kernel
':%2conv2d_transpose_15/bias
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
Е
layer_metrics
layers
metrics
non_trainable_variables
 layer_regularization_losses
:trainable_variables
;regularization_losses
<	variables
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layer_metrics
layers
metrics
non_trainable_variables
 layer_regularization_losses
>trainable_variables
?regularization_losses
@	variables
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
6:42conv2d_transpose_16/kernel
':%2conv2d_transpose_16/bias
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
Е
layer_metrics
layers
metrics
non_trainable_variables
 layer_regularization_losses
Dtrainable_variables
Eregularization_losses
F	variables
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layer_metrics
layers
metrics
non_trainable_variables
 layer_regularization_losses
Htrainable_variables
Iregularization_losses
J	variables
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
6:42conv2d_transpose_17/kernel
':%2conv2d_transpose_17/bias
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
Е
layer_metrics
layers
metrics
non_trainable_variables
 layer_regularization_losses
Ntrainable_variables
Oregularization_losses
P	variables
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layer_metrics
layers
metrics
 non_trainable_variables
 Ёlayer_regularization_losses
Rtrainable_variables
Sregularization_losses
T	variables
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_15/kernel
:2conv2d_15/bias
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
Е
Ђlayer_metrics
Ѓlayers
Єmetrics
Ѕnon_trainable_variables
 Іlayer_regularization_losses
Xtrainable_variables
Yregularization_losses
Z	variables
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
њ2ї
+__inference_model_13_layer_call_fn_15573090
+__inference_model_13_layer_call_fn_15573058
+__inference_model_13_layer_call_fn_15572649
+__inference_model_13_layer_call_fn_15572726Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
#__inference__wrapped_model_15572140р
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *PЂM
KH
"
input_21џџџџџџџџџd
"
input_22џџџџџџџџџ
ц2у
F__inference_model_13_layer_call_and_return_conditional_losses_15572893
F__inference_model_13_layer_call_and_return_conditional_losses_15572526
F__inference_model_13_layer_call_and_return_conditional_losses_15573026
F__inference_model_13_layer_call_and_return_conditional_losses_15572571Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
е2в
+__inference_dense_25_layer_call_fn_15573109Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_25_layer_call_and_return_conditional_losses_15573100Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
й2ж
/__inference_embedding_10_layer_call_fn_15573126Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
є2ё
J__inference_embedding_10_layer_call_and_return_conditional_losses_15573119Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_leaky_re_lu_35_layer_call_fn_15573136Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_leaky_re_lu_35_layer_call_and_return_conditional_losses_15573131Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_26_layer_call_fn_15573175Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_26_layer_call_and_return_conditional_losses_15573166Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_reshape_15_layer_call_fn_15573194Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_reshape_15_layer_call_and_return_conditional_losses_15573189Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_reshape_16_layer_call_fn_15573213Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_reshape_16_layer_call_and_return_conditional_losses_15573208Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_concatenate_10_layer_call_fn_15573226Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_concatenate_10_layer_call_and_return_conditional_losses_15573220Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
6__inference_conv2d_transpose_15_layer_call_fn_15572184и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Б2Ў
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_15572174и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
л2и
1__inference_leaky_re_lu_36_layer_call_fn_15573236Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_leaky_re_lu_36_layer_call_and_return_conditional_losses_15573231Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
6__inference_conv2d_transpose_16_layer_call_fn_15572228и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Б2Ў
Q__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_15572218и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
л2и
1__inference_leaky_re_lu_37_layer_call_fn_15573246Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_leaky_re_lu_37_layer_call_and_return_conditional_losses_15573241Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
6__inference_conv2d_transpose_17_layer_call_fn_15572272и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Б2Ў
Q__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_15572262и
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *8Ђ5
30,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
л2и
1__inference_leaky_re_lu_38_layer_call_fn_15573256Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_leaky_re_lu_38_layer_call_and_return_conditional_losses_15573251Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_conv2d_15_layer_call_fn_15573276Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_conv2d_15_layer_call_and_return_conditional_losses_15573267Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
жBг
&__inference_signature_wrapper_15572760input_21input_22"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 в
#__inference__wrapped_model_15572140Њ&'89BCLMVWZЂW
PЂM
KH
"
input_21џџџџџџџџџd
"
input_22џџџџџџџџџ
Њ "=Њ:
8
	conv2d_15+(
	conv2d_15џџџџџџџџџ``ю
L__inference_concatenate_10_layer_call_and_return_conditional_losses_15573220kЂh
aЂ^
\Y
+(
inputs/0џџџџџџџџџ
*'
inputs/1џџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 Ц
1__inference_concatenate_10_layer_call_fn_15573226kЂh
aЂ^
\Y
+(
inputs/0џџџџџџџџџ
*'
inputs/1џџџџџџџџџ
Њ "!џџџџџџџџџн
G__inference_conv2d_15_layer_call_and_return_conditional_losses_15573267VWJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Е
,__inference_conv2d_15_layer_call_fn_15573276VWJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџш
Q__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_1557217489JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Р
6__inference_conv2d_transpose_15_layer_call_fn_1557218489JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџш
Q__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_15572218BCJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Р
6__inference_conv2d_transpose_16_layer_call_fn_15572228BCJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџш
Q__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_15572262LMJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Р
6__inference_conv2d_transpose_17_layer_call_fn_15572272LMJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџЈ
F__inference_dense_25_layer_call_and_return_conditional_losses_15573100^/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "'Ђ$

0џџџџџџџџџ
 
+__inference_dense_25_layer_call_fn_15573109Q/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "џџџџџџџџџЏ
F__inference_dense_26_layer_call_and_return_conditional_losses_15573166e&'3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ2
Њ "*Ђ'
 
0џџџџџџџџџА
 
+__inference_dense_26_layer_call_fn_15573175X&'3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ2
Њ "џџџџџџџџџА­
J__inference_embedding_10_layer_call_and_return_conditional_losses_15573119_/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ2
 
/__inference_embedding_10_layer_call_fn_15573126R/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ2Ќ
L__inference_leaky_re_lu_35_layer_call_and_return_conditional_losses_15573131\1Ђ.
'Ђ$
"
inputsџџџџџџџџџ
Њ "'Ђ$

0џџџџџџџџџ
 
1__inference_leaky_re_lu_35_layer_call_fn_15573136O1Ђ.
'Ђ$
"
inputsџџџџџџџџџ
Њ "џџџџџџџџџп
L__inference_leaky_re_lu_36_layer_call_and_return_conditional_losses_15573231JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
1__inference_leaky_re_lu_36_layer_call_fn_15573236JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџп
L__inference_leaky_re_lu_37_layer_call_and_return_conditional_losses_15573241JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
1__inference_leaky_re_lu_37_layer_call_fn_15573246JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџп
L__inference_leaky_re_lu_38_layer_call_and_return_conditional_losses_15573251JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
1__inference_leaky_re_lu_38_layer_call_fn_15573256JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџџ
F__inference_model_13_layer_call_and_return_conditional_losses_15572526Д&'89BCLMVWbЂ_
XЂU
KH
"
input_21џџџџџџџџџd
"
input_22џџџџџџџџџ
p

 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 џ
F__inference_model_13_layer_call_and_return_conditional_losses_15572571Д&'89BCLMVWbЂ_
XЂU
KH
"
input_21џџџџџџџџџd
"
input_22џџџџџџџџџ
p 

 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 э
F__inference_model_13_layer_call_and_return_conditional_losses_15572893Ђ&'89BCLMVWbЂ_
XЂU
KH
"
inputs/0џџџџџџџџџd
"
inputs/1џџџџџџџџџ
p

 
Њ "-Ђ*
# 
0џџџџџџџџџ``
 э
F__inference_model_13_layer_call_and_return_conditional_losses_15573026Ђ&'89BCLMVWbЂ_
XЂU
KH
"
inputs/0џџџџџџџџџd
"
inputs/1џџџџџџџџџ
p 

 
Њ "-Ђ*
# 
0џџџџџџџџџ``
 з
+__inference_model_13_layer_call_fn_15572649Ї&'89BCLMVWbЂ_
XЂU
KH
"
input_21џџџџџџџџџd
"
input_22џџџџџџџџџ
p

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџз
+__inference_model_13_layer_call_fn_15572726Ї&'89BCLMVWbЂ_
XЂU
KH
"
input_21џџџџџџџџџd
"
input_22џџџџџџџџџ
p 

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџз
+__inference_model_13_layer_call_fn_15573058Ї&'89BCLMVWbЂ_
XЂU
KH
"
inputs/0џџџџџџџџџd
"
inputs/1џџџџџџџџџ
p

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџз
+__inference_model_13_layer_call_fn_15573090Ї&'89BCLMVWbЂ_
XЂU
KH
"
inputs/0џџџџџџџџџd
"
inputs/1џџџџџџџџџ
p 

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЏ
H__inference_reshape_15_layer_call_and_return_conditional_losses_15573189c1Ђ.
'Ђ$
"
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
-__inference_reshape_15_layer_call_fn_15573194V1Ђ.
'Ђ$
"
inputsџџџџџџџџџ
Њ "!џџџџџџџџџБ
H__inference_reshape_16_layer_call_and_return_conditional_losses_15573208e4Ђ1
*Ђ'
%"
inputsџџџџџџџџџА
Њ "-Ђ*
# 
0џџџџџџџџџ
 
-__inference_reshape_16_layer_call_fn_15573213X4Ђ1
*Ђ'
%"
inputsџџџџџџџџџА
Њ " џџџџџџџџџш
&__inference_signature_wrapper_15572760Н&'89BCLMVWmЂj
Ђ 
cЊ`
.
input_21"
input_21џџџџџџџџџd
.
input_22"
input_22џџџџџџџџџ"=Њ:
8
	conv2d_15+(
	conv2d_15џџџџџџџџџ``