ϫ	
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�"serve*2.16.12v2.16.0-rc0-18-g5bc9d26649c8��
j
ConstConst*&
_output_shapes
:*
dtype0*%
valueB*�Y4?
l
Const_1Const*&
_output_shapes
:*
dtype0*%
valueB*(>
�
normalization/countVarHandleOp*
_output_shapes
: *$

debug_namenormalization/count/*
dtype0	*
shape: *$
shared_namenormalization/count
s
'normalization/count/Read/ReadVariableOpReadVariableOpnormalization/count*
_output_shapes
: *
dtype0	
�
#Variable/Initializer/ReadVariableOpReadVariableOpnormalization/count*
_class
loc:@Variable*
_output_shapes
: *
dtype0	
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0	*
shape: *
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0	
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
�
normalization/varianceVarHandleOp*
_output_shapes
: *'

debug_namenormalization/variance/*
dtype0*
shape:*'
shared_namenormalization/variance
}
*normalization/variance/Read/ReadVariableOpReadVariableOpnormalization/variance*
_output_shapes
:*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOpnormalization/variance*
_class
loc:@Variable_1*
_output_shapes
:*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape:*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
e
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:*
dtype0
�
normalization/meanVarHandleOp*
_output_shapes
: *#

debug_namenormalization/mean/*
dtype0*
shape:*#
shared_namenormalization/mean
u
&normalization/mean/Read/ReadVariableOpReadVariableOpnormalization/mean*
_output_shapes
:*
dtype0
�
%Variable_2/Initializer/ReadVariableOpReadVariableOpnormalization/mean*
_class
loc:@Variable_2*
_output_shapes
:*
dtype0
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape:*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:*
dtype0
�
sequential/dense_1/biasVarHandleOp*
_output_shapes
: *(

debug_namesequential/dense_1/bias/*
dtype0*
shape:*(
shared_namesequential/dense_1/bias

+sequential/dense_1/bias/Read/ReadVariableOpReadVariableOpsequential/dense_1/bias*
_output_shapes
:*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOpsequential/dense_1/bias*
_class
loc:@Variable_3*
_output_shapes
:*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape:*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
e
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
:*
dtype0
�
sequential/dense_1/kernelVarHandleOp*
_output_shapes
: **

debug_namesequential/dense_1/kernel/*
dtype0*
shape:	�**
shared_namesequential/dense_1/kernel
�
-sequential/dense_1/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_1/kernel*
_output_shapes
:	�*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOpsequential/dense_1/kernel*
_class
loc:@Variable_4*
_output_shapes
:	�*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:	�*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
j
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
:	�*
dtype0
�
sequential/dense/biasVarHandleOp*
_output_shapes
: *&

debug_namesequential/dense/bias/*
dtype0*
shape:�*&
shared_namesequential/dense/bias
|
)sequential/dense/bias/Read/ReadVariableOpReadVariableOpsequential/dense/bias*
_output_shapes	
:�*
dtype0
�
%Variable_5/Initializer/ReadVariableOpReadVariableOpsequential/dense/bias*
_class
loc:@Variable_5*
_output_shapes	
:�*
dtype0
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape:�*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0
f
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes	
:�*
dtype0
�
sequential/dense/kernelVarHandleOp*
_output_shapes
: *(

debug_namesequential/dense/kernel/*
dtype0*
shape:
�b�*(
shared_namesequential/dense/kernel
�
+sequential/dense/kernel/Read/ReadVariableOpReadVariableOpsequential/dense/kernel* 
_output_shapes
:
�b�*
dtype0
�
%Variable_6/Initializer/ReadVariableOpReadVariableOpsequential/dense/kernel*
_class
loc:@Variable_6* 
_output_shapes
:
�b�*
dtype0
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape:
�b�*
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0
k
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6* 
_output_shapes
:
�b�*
dtype0
�
sequential/conv2d_1/biasVarHandleOp*
_output_shapes
: *)

debug_namesequential/conv2d_1/bias/*
dtype0*
shape:@*)
shared_namesequential/conv2d_1/bias
�
,sequential/conv2d_1/bias/Read/ReadVariableOpReadVariableOpsequential/conv2d_1/bias*
_output_shapes
:@*
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOpsequential/conv2d_1/bias*
_class
loc:@Variable_7*
_output_shapes
:@*
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape:@*
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
e
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
_output_shapes
:@*
dtype0
�
sequential/conv2d_1/kernelVarHandleOp*
_output_shapes
: *+

debug_namesequential/conv2d_1/kernel/*
dtype0*
shape: @*+
shared_namesequential/conv2d_1/kernel
�
.sequential/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpsequential/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
�
%Variable_8/Initializer/ReadVariableOpReadVariableOpsequential/conv2d_1/kernel*
_class
loc:@Variable_8*&
_output_shapes
: @*
dtype0
�

Variable_8VarHandleOp*
_class
loc:@Variable_8*
_output_shapes
: *

debug_nameVariable_8/*
dtype0*
shape: @*
shared_name
Variable_8
e
+Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_8*
_output_shapes
: 
e
Variable_8/AssignAssignVariableOp
Variable_8%Variable_8/Initializer/ReadVariableOp*
dtype0
q
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*&
_output_shapes
: @*
dtype0
�
sequential/conv2d/biasVarHandleOp*
_output_shapes
: *'

debug_namesequential/conv2d/bias/*
dtype0*
shape: *'
shared_namesequential/conv2d/bias
}
*sequential/conv2d/bias/Read/ReadVariableOpReadVariableOpsequential/conv2d/bias*
_output_shapes
: *
dtype0
�
%Variable_9/Initializer/ReadVariableOpReadVariableOpsequential/conv2d/bias*
_class
loc:@Variable_9*
_output_shapes
: *
dtype0
�

Variable_9VarHandleOp*
_class
loc:@Variable_9*
_output_shapes
: *

debug_nameVariable_9/*
dtype0*
shape: *
shared_name
Variable_9
e
+Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_9*
_output_shapes
: 
e
Variable_9/AssignAssignVariableOp
Variable_9%Variable_9/Initializer/ReadVariableOp*
dtype0
e
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes
: *
dtype0
�
sequential/conv2d/kernelVarHandleOp*
_output_shapes
: *)

debug_namesequential/conv2d/kernel/*
dtype0*
shape: *)
shared_namesequential/conv2d/kernel
�
,sequential/conv2d/kernel/Read/ReadVariableOpReadVariableOpsequential/conv2d/kernel*&
_output_shapes
: *
dtype0
�
&Variable_10/Initializer/ReadVariableOpReadVariableOpsequential/conv2d/kernel*
_class
loc:@Variable_10*&
_output_shapes
: *
dtype0
�
Variable_10VarHandleOp*
_class
loc:@Variable_10*
_output_shapes
: *

debug_nameVariable_10/*
dtype0*
shape: *
shared_nameVariable_10
g
,Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_10*
_output_shapes
: 
h
Variable_10/AssignAssignVariableOpVariable_10&Variable_10/Initializer/ReadVariableOp*
dtype0
s
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*&
_output_shapes
: *
dtype0
�
%adam/sequential_dense_1_bias_velocityVarHandleOp*
_output_shapes
: *6

debug_name(&adam/sequential_dense_1_bias_velocity/*
dtype0*
shape:*6
shared_name'%adam/sequential_dense_1_bias_velocity
�
9adam/sequential_dense_1_bias_velocity/Read/ReadVariableOpReadVariableOp%adam/sequential_dense_1_bias_velocity*
_output_shapes
:*
dtype0
�
&Variable_11/Initializer/ReadVariableOpReadVariableOp%adam/sequential_dense_1_bias_velocity*
_class
loc:@Variable_11*
_output_shapes
:*
dtype0
�
Variable_11VarHandleOp*
_class
loc:@Variable_11*
_output_shapes
: *

debug_nameVariable_11/*
dtype0*
shape:*
shared_nameVariable_11
g
,Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_11*
_output_shapes
: 
h
Variable_11/AssignAssignVariableOpVariable_11&Variable_11/Initializer/ReadVariableOp*
dtype0
g
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes
:*
dtype0
�
%adam/sequential_dense_1_bias_momentumVarHandleOp*
_output_shapes
: *6

debug_name(&adam/sequential_dense_1_bias_momentum/*
dtype0*
shape:*6
shared_name'%adam/sequential_dense_1_bias_momentum
�
9adam/sequential_dense_1_bias_momentum/Read/ReadVariableOpReadVariableOp%adam/sequential_dense_1_bias_momentum*
_output_shapes
:*
dtype0
�
&Variable_12/Initializer/ReadVariableOpReadVariableOp%adam/sequential_dense_1_bias_momentum*
_class
loc:@Variable_12*
_output_shapes
:*
dtype0
�
Variable_12VarHandleOp*
_class
loc:@Variable_12*
_output_shapes
: *

debug_nameVariable_12/*
dtype0*
shape:*
shared_nameVariable_12
g
,Variable_12/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_12*
_output_shapes
: 
h
Variable_12/AssignAssignVariableOpVariable_12&Variable_12/Initializer/ReadVariableOp*
dtype0
g
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12*
_output_shapes
:*
dtype0
�
'adam/sequential_dense_1_kernel_velocityVarHandleOp*
_output_shapes
: *8

debug_name*(adam/sequential_dense_1_kernel_velocity/*
dtype0*
shape:	�*8
shared_name)'adam/sequential_dense_1_kernel_velocity
�
;adam/sequential_dense_1_kernel_velocity/Read/ReadVariableOpReadVariableOp'adam/sequential_dense_1_kernel_velocity*
_output_shapes
:	�*
dtype0
�
&Variable_13/Initializer/ReadVariableOpReadVariableOp'adam/sequential_dense_1_kernel_velocity*
_class
loc:@Variable_13*
_output_shapes
:	�*
dtype0
�
Variable_13VarHandleOp*
_class
loc:@Variable_13*
_output_shapes
: *

debug_nameVariable_13/*
dtype0*
shape:	�*
shared_nameVariable_13
g
,Variable_13/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_13*
_output_shapes
: 
h
Variable_13/AssignAssignVariableOpVariable_13&Variable_13/Initializer/ReadVariableOp*
dtype0
l
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*
_output_shapes
:	�*
dtype0
�
'adam/sequential_dense_1_kernel_momentumVarHandleOp*
_output_shapes
: *8

debug_name*(adam/sequential_dense_1_kernel_momentum/*
dtype0*
shape:	�*8
shared_name)'adam/sequential_dense_1_kernel_momentum
�
;adam/sequential_dense_1_kernel_momentum/Read/ReadVariableOpReadVariableOp'adam/sequential_dense_1_kernel_momentum*
_output_shapes
:	�*
dtype0
�
&Variable_14/Initializer/ReadVariableOpReadVariableOp'adam/sequential_dense_1_kernel_momentum*
_class
loc:@Variable_14*
_output_shapes
:	�*
dtype0
�
Variable_14VarHandleOp*
_class
loc:@Variable_14*
_output_shapes
: *

debug_nameVariable_14/*
dtype0*
shape:	�*
shared_nameVariable_14
g
,Variable_14/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_14*
_output_shapes
: 
h
Variable_14/AssignAssignVariableOpVariable_14&Variable_14/Initializer/ReadVariableOp*
dtype0
l
Variable_14/Read/ReadVariableOpReadVariableOpVariable_14*
_output_shapes
:	�*
dtype0
�
#adam/sequential_dense_bias_velocityVarHandleOp*
_output_shapes
: *4

debug_name&$adam/sequential_dense_bias_velocity/*
dtype0*
shape:�*4
shared_name%#adam/sequential_dense_bias_velocity
�
7adam/sequential_dense_bias_velocity/Read/ReadVariableOpReadVariableOp#adam/sequential_dense_bias_velocity*
_output_shapes	
:�*
dtype0
�
&Variable_15/Initializer/ReadVariableOpReadVariableOp#adam/sequential_dense_bias_velocity*
_class
loc:@Variable_15*
_output_shapes	
:�*
dtype0
�
Variable_15VarHandleOp*
_class
loc:@Variable_15*
_output_shapes
: *

debug_nameVariable_15/*
dtype0*
shape:�*
shared_nameVariable_15
g
,Variable_15/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_15*
_output_shapes
: 
h
Variable_15/AssignAssignVariableOpVariable_15&Variable_15/Initializer/ReadVariableOp*
dtype0
h
Variable_15/Read/ReadVariableOpReadVariableOpVariable_15*
_output_shapes	
:�*
dtype0
�
#adam/sequential_dense_bias_momentumVarHandleOp*
_output_shapes
: *4

debug_name&$adam/sequential_dense_bias_momentum/*
dtype0*
shape:�*4
shared_name%#adam/sequential_dense_bias_momentum
�
7adam/sequential_dense_bias_momentum/Read/ReadVariableOpReadVariableOp#adam/sequential_dense_bias_momentum*
_output_shapes	
:�*
dtype0
�
&Variable_16/Initializer/ReadVariableOpReadVariableOp#adam/sequential_dense_bias_momentum*
_class
loc:@Variable_16*
_output_shapes	
:�*
dtype0
�
Variable_16VarHandleOp*
_class
loc:@Variable_16*
_output_shapes
: *

debug_nameVariable_16/*
dtype0*
shape:�*
shared_nameVariable_16
g
,Variable_16/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_16*
_output_shapes
: 
h
Variable_16/AssignAssignVariableOpVariable_16&Variable_16/Initializer/ReadVariableOp*
dtype0
h
Variable_16/Read/ReadVariableOpReadVariableOpVariable_16*
_output_shapes	
:�*
dtype0
�
%adam/sequential_dense_kernel_velocityVarHandleOp*
_output_shapes
: *6

debug_name(&adam/sequential_dense_kernel_velocity/*
dtype0*
shape:
�b�*6
shared_name'%adam/sequential_dense_kernel_velocity
�
9adam/sequential_dense_kernel_velocity/Read/ReadVariableOpReadVariableOp%adam/sequential_dense_kernel_velocity* 
_output_shapes
:
�b�*
dtype0
�
&Variable_17/Initializer/ReadVariableOpReadVariableOp%adam/sequential_dense_kernel_velocity*
_class
loc:@Variable_17* 
_output_shapes
:
�b�*
dtype0
�
Variable_17VarHandleOp*
_class
loc:@Variable_17*
_output_shapes
: *

debug_nameVariable_17/*
dtype0*
shape:
�b�*
shared_nameVariable_17
g
,Variable_17/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_17*
_output_shapes
: 
h
Variable_17/AssignAssignVariableOpVariable_17&Variable_17/Initializer/ReadVariableOp*
dtype0
m
Variable_17/Read/ReadVariableOpReadVariableOpVariable_17* 
_output_shapes
:
�b�*
dtype0
�
%adam/sequential_dense_kernel_momentumVarHandleOp*
_output_shapes
: *6

debug_name(&adam/sequential_dense_kernel_momentum/*
dtype0*
shape:
�b�*6
shared_name'%adam/sequential_dense_kernel_momentum
�
9adam/sequential_dense_kernel_momentum/Read/ReadVariableOpReadVariableOp%adam/sequential_dense_kernel_momentum* 
_output_shapes
:
�b�*
dtype0
�
&Variable_18/Initializer/ReadVariableOpReadVariableOp%adam/sequential_dense_kernel_momentum*
_class
loc:@Variable_18* 
_output_shapes
:
�b�*
dtype0
�
Variable_18VarHandleOp*
_class
loc:@Variable_18*
_output_shapes
: *

debug_nameVariable_18/*
dtype0*
shape:
�b�*
shared_nameVariable_18
g
,Variable_18/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_18*
_output_shapes
: 
h
Variable_18/AssignAssignVariableOpVariable_18&Variable_18/Initializer/ReadVariableOp*
dtype0
m
Variable_18/Read/ReadVariableOpReadVariableOpVariable_18* 
_output_shapes
:
�b�*
dtype0
�
&adam/sequential_conv2d_1_bias_velocityVarHandleOp*
_output_shapes
: *7

debug_name)'adam/sequential_conv2d_1_bias_velocity/*
dtype0*
shape:@*7
shared_name(&adam/sequential_conv2d_1_bias_velocity
�
:adam/sequential_conv2d_1_bias_velocity/Read/ReadVariableOpReadVariableOp&adam/sequential_conv2d_1_bias_velocity*
_output_shapes
:@*
dtype0
�
&Variable_19/Initializer/ReadVariableOpReadVariableOp&adam/sequential_conv2d_1_bias_velocity*
_class
loc:@Variable_19*
_output_shapes
:@*
dtype0
�
Variable_19VarHandleOp*
_class
loc:@Variable_19*
_output_shapes
: *

debug_nameVariable_19/*
dtype0*
shape:@*
shared_nameVariable_19
g
,Variable_19/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_19*
_output_shapes
: 
h
Variable_19/AssignAssignVariableOpVariable_19&Variable_19/Initializer/ReadVariableOp*
dtype0
g
Variable_19/Read/ReadVariableOpReadVariableOpVariable_19*
_output_shapes
:@*
dtype0
�
&adam/sequential_conv2d_1_bias_momentumVarHandleOp*
_output_shapes
: *7

debug_name)'adam/sequential_conv2d_1_bias_momentum/*
dtype0*
shape:@*7
shared_name(&adam/sequential_conv2d_1_bias_momentum
�
:adam/sequential_conv2d_1_bias_momentum/Read/ReadVariableOpReadVariableOp&adam/sequential_conv2d_1_bias_momentum*
_output_shapes
:@*
dtype0
�
&Variable_20/Initializer/ReadVariableOpReadVariableOp&adam/sequential_conv2d_1_bias_momentum*
_class
loc:@Variable_20*
_output_shapes
:@*
dtype0
�
Variable_20VarHandleOp*
_class
loc:@Variable_20*
_output_shapes
: *

debug_nameVariable_20/*
dtype0*
shape:@*
shared_nameVariable_20
g
,Variable_20/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_20*
_output_shapes
: 
h
Variable_20/AssignAssignVariableOpVariable_20&Variable_20/Initializer/ReadVariableOp*
dtype0
g
Variable_20/Read/ReadVariableOpReadVariableOpVariable_20*
_output_shapes
:@*
dtype0
�
(adam/sequential_conv2d_1_kernel_velocityVarHandleOp*
_output_shapes
: *9

debug_name+)adam/sequential_conv2d_1_kernel_velocity/*
dtype0*
shape: @*9
shared_name*(adam/sequential_conv2d_1_kernel_velocity
�
<adam/sequential_conv2d_1_kernel_velocity/Read/ReadVariableOpReadVariableOp(adam/sequential_conv2d_1_kernel_velocity*&
_output_shapes
: @*
dtype0
�
&Variable_21/Initializer/ReadVariableOpReadVariableOp(adam/sequential_conv2d_1_kernel_velocity*
_class
loc:@Variable_21*&
_output_shapes
: @*
dtype0
�
Variable_21VarHandleOp*
_class
loc:@Variable_21*
_output_shapes
: *

debug_nameVariable_21/*
dtype0*
shape: @*
shared_nameVariable_21
g
,Variable_21/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_21*
_output_shapes
: 
h
Variable_21/AssignAssignVariableOpVariable_21&Variable_21/Initializer/ReadVariableOp*
dtype0
s
Variable_21/Read/ReadVariableOpReadVariableOpVariable_21*&
_output_shapes
: @*
dtype0
�
(adam/sequential_conv2d_1_kernel_momentumVarHandleOp*
_output_shapes
: *9

debug_name+)adam/sequential_conv2d_1_kernel_momentum/*
dtype0*
shape: @*9
shared_name*(adam/sequential_conv2d_1_kernel_momentum
�
<adam/sequential_conv2d_1_kernel_momentum/Read/ReadVariableOpReadVariableOp(adam/sequential_conv2d_1_kernel_momentum*&
_output_shapes
: @*
dtype0
�
&Variable_22/Initializer/ReadVariableOpReadVariableOp(adam/sequential_conv2d_1_kernel_momentum*
_class
loc:@Variable_22*&
_output_shapes
: @*
dtype0
�
Variable_22VarHandleOp*
_class
loc:@Variable_22*
_output_shapes
: *

debug_nameVariable_22/*
dtype0*
shape: @*
shared_nameVariable_22
g
,Variable_22/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_22*
_output_shapes
: 
h
Variable_22/AssignAssignVariableOpVariable_22&Variable_22/Initializer/ReadVariableOp*
dtype0
s
Variable_22/Read/ReadVariableOpReadVariableOpVariable_22*&
_output_shapes
: @*
dtype0
�
$adam/sequential_conv2d_bias_velocityVarHandleOp*
_output_shapes
: *5

debug_name'%adam/sequential_conv2d_bias_velocity/*
dtype0*
shape: *5
shared_name&$adam/sequential_conv2d_bias_velocity
�
8adam/sequential_conv2d_bias_velocity/Read/ReadVariableOpReadVariableOp$adam/sequential_conv2d_bias_velocity*
_output_shapes
: *
dtype0
�
&Variable_23/Initializer/ReadVariableOpReadVariableOp$adam/sequential_conv2d_bias_velocity*
_class
loc:@Variable_23*
_output_shapes
: *
dtype0
�
Variable_23VarHandleOp*
_class
loc:@Variable_23*
_output_shapes
: *

debug_nameVariable_23/*
dtype0*
shape: *
shared_nameVariable_23
g
,Variable_23/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_23*
_output_shapes
: 
h
Variable_23/AssignAssignVariableOpVariable_23&Variable_23/Initializer/ReadVariableOp*
dtype0
g
Variable_23/Read/ReadVariableOpReadVariableOpVariable_23*
_output_shapes
: *
dtype0
�
$adam/sequential_conv2d_bias_momentumVarHandleOp*
_output_shapes
: *5

debug_name'%adam/sequential_conv2d_bias_momentum/*
dtype0*
shape: *5
shared_name&$adam/sequential_conv2d_bias_momentum
�
8adam/sequential_conv2d_bias_momentum/Read/ReadVariableOpReadVariableOp$adam/sequential_conv2d_bias_momentum*
_output_shapes
: *
dtype0
�
&Variable_24/Initializer/ReadVariableOpReadVariableOp$adam/sequential_conv2d_bias_momentum*
_class
loc:@Variable_24*
_output_shapes
: *
dtype0
�
Variable_24VarHandleOp*
_class
loc:@Variable_24*
_output_shapes
: *

debug_nameVariable_24/*
dtype0*
shape: *
shared_nameVariable_24
g
,Variable_24/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_24*
_output_shapes
: 
h
Variable_24/AssignAssignVariableOpVariable_24&Variable_24/Initializer/ReadVariableOp*
dtype0
g
Variable_24/Read/ReadVariableOpReadVariableOpVariable_24*
_output_shapes
: *
dtype0
�
&adam/sequential_conv2d_kernel_velocityVarHandleOp*
_output_shapes
: *7

debug_name)'adam/sequential_conv2d_kernel_velocity/*
dtype0*
shape: *7
shared_name(&adam/sequential_conv2d_kernel_velocity
�
:adam/sequential_conv2d_kernel_velocity/Read/ReadVariableOpReadVariableOp&adam/sequential_conv2d_kernel_velocity*&
_output_shapes
: *
dtype0
�
&Variable_25/Initializer/ReadVariableOpReadVariableOp&adam/sequential_conv2d_kernel_velocity*
_class
loc:@Variable_25*&
_output_shapes
: *
dtype0
�
Variable_25VarHandleOp*
_class
loc:@Variable_25*
_output_shapes
: *

debug_nameVariable_25/*
dtype0*
shape: *
shared_nameVariable_25
g
,Variable_25/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_25*
_output_shapes
: 
h
Variable_25/AssignAssignVariableOpVariable_25&Variable_25/Initializer/ReadVariableOp*
dtype0
s
Variable_25/Read/ReadVariableOpReadVariableOpVariable_25*&
_output_shapes
: *
dtype0
�
&adam/sequential_conv2d_kernel_momentumVarHandleOp*
_output_shapes
: *7

debug_name)'adam/sequential_conv2d_kernel_momentum/*
dtype0*
shape: *7
shared_name(&adam/sequential_conv2d_kernel_momentum
�
:adam/sequential_conv2d_kernel_momentum/Read/ReadVariableOpReadVariableOp&adam/sequential_conv2d_kernel_momentum*&
_output_shapes
: *
dtype0
�
&Variable_26/Initializer/ReadVariableOpReadVariableOp&adam/sequential_conv2d_kernel_momentum*
_class
loc:@Variable_26*&
_output_shapes
: *
dtype0
�
Variable_26VarHandleOp*
_class
loc:@Variable_26*
_output_shapes
: *

debug_nameVariable_26/*
dtype0*
shape: *
shared_nameVariable_26
g
,Variable_26/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_26*
_output_shapes
: 
h
Variable_26/AssignAssignVariableOpVariable_26&Variable_26/Initializer/ReadVariableOp*
dtype0
s
Variable_26/Read/ReadVariableOpReadVariableOpVariable_26*&
_output_shapes
: *
dtype0
�
adam/learning_rateVarHandleOp*
_output_shapes
: *#

debug_nameadam/learning_rate/*
dtype0*
shape: *#
shared_nameadam/learning_rate
q
&adam/learning_rate/Read/ReadVariableOpReadVariableOpadam/learning_rate*
_output_shapes
: *
dtype0
�
&Variable_27/Initializer/ReadVariableOpReadVariableOpadam/learning_rate*
_class
loc:@Variable_27*
_output_shapes
: *
dtype0
�
Variable_27VarHandleOp*
_class
loc:@Variable_27*
_output_shapes
: *

debug_nameVariable_27/*
dtype0*
shape: *
shared_nameVariable_27
g
,Variable_27/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_27*
_output_shapes
: 
h
Variable_27/AssignAssignVariableOpVariable_27&Variable_27/Initializer/ReadVariableOp*
dtype0
c
Variable_27/Read/ReadVariableOpReadVariableOpVariable_27*
_output_shapes
: *
dtype0
�
adam/iterationVarHandleOp*
_output_shapes
: *

debug_nameadam/iteration/*
dtype0	*
shape: *
shared_nameadam/iteration
i
"adam/iteration/Read/ReadVariableOpReadVariableOpadam/iteration*
_output_shapes
: *
dtype0	
�
&Variable_28/Initializer/ReadVariableOpReadVariableOpadam/iteration*
_class
loc:@Variable_28*
_output_shapes
: *
dtype0	
�
Variable_28VarHandleOp*
_class
loc:@Variable_28*
_output_shapes
: *

debug_nameVariable_28/*
dtype0	*
shape: *
shared_nameVariable_28
g
,Variable_28/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_28*
_output_shapes
: 
h
Variable_28/AssignAssignVariableOpVariable_28&Variable_28/Initializer/ReadVariableOp*
dtype0	
c
Variable_28/Read/ReadVariableOpReadVariableOpVariable_28*
_output_shapes
: *
dtype0	

NoOpNoOp
�.
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*�.
value�.B�. B�-
-
	model
__call__

signatures*
�
_functional
	optimizer
_default_save_signature
_inbound_nodes
_outbound_nodes
	_losses

	_loss_ids
_losses_override
_layers
_build_shapes_dict*

trace_0
trace_1* 
* 
�
_tracked
_inbound_nodes
_outbound_nodes
_losses
_losses_override
_operations
_layers
_build_shapes_dict
output_names
_default_save_signature*
�

_variables
_trainable_variables
 _trainable_variables_indices

iterations
_learning_rate

_momentums
 _velocities*

!trace_0* 
* 
* 
* 
* 
* 
R
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10*
* 
 
-	capture_0
.	capture_1* 
 
-	capture_0
.	capture_1* 
* 
* 
* 
* 
* 
R
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10*
R
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10*
* 
* 

/trace_0* 
�
0
1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15
>16
?17*
<
@0
A1
B2
C3
D4
E5
F6
G7*
* 
ZT
VARIABLE_VALUEVariable_285model/optimizer/iterations/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEVariable_279model/optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
 
-	capture_0
.	capture_1* 
]
H_inbound_nodes
I_outbound_nodes
J_losses
K	_loss_ids
L_losses_override* 
u
M_inbound_nodes
N_outbound_nodes
O_losses
P	_loss_ids
Q_losses_override
R_build_shapes_dict* 
�
S
adapt_mean
Tadapt_variance
	Ucount
V_inbound_nodes
W_outbound_nodes
X_losses
Y	_loss_ids
Z_losses_override
[_reduce_axis_mask
\_broadcast_shape
]_build_shapes_dict*
�
@_kernel
Abias
^_inbound_nodes
__outbound_nodes
`_losses
a	_loss_ids
b_losses_override
c_build_shapes_dict*
�
B_kernel
Cbias
d_inbound_nodes
e_outbound_nodes
f_losses
g	_loss_ids
h_losses_override
i_build_shapes_dict*
u
j_inbound_nodes
k_outbound_nodes
l_losses
m	_loss_ids
n_losses_override
o_build_shapes_dict* 
u
p_inbound_nodes
q_outbound_nodes
r_losses
s	_loss_ids
t_losses_override
u_build_shapes_dict* 
u
v_inbound_nodes
w_outbound_nodes
x_losses
y	_loss_ids
z_losses_override
{_build_shapes_dict* 
�
D_kernel
Ebias
|_inbound_nodes
}_outbound_nodes
~_losses
	_loss_ids
�_losses_override
�_build_shapes_dict*
{
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids
�_losses_override
�_build_shapes_dict* 
�
F_kernel
Gbias
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids
�_losses_override
�_build_shapes_dict*
* 
* 
 
-	capture_0
.	capture_1* 
\V
VARIABLE_VALUEVariable_267model/optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEVariable_257model/optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEVariable_247model/optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEVariable_237model/optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEVariable_227model/optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEVariable_217model/optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEVariable_207model/optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEVariable_197model/optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEVariable_188model/optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEVariable_178model/optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEVariable_168model/optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEVariable_158model/optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEVariable_148model/optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEVariable_138model/optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEVariable_128model/optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEVariable_118model/optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEVariable_10Amodel/optimizer/_trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE
Variable_9Amodel/optimizer/_trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE
Variable_8Amodel/optimizer/_trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE
Variable_7Amodel/optimizer/_trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE
Variable_6Amodel/optimizer/_trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE
Variable_5Amodel/optimizer/_trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE
Variable_4Amodel/optimizer/_trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE
Variable_3Amodel/optimizer/_trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
YS
VARIABLE_VALUE
Variable_25model/_layers/2/adapt_mean/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUE
Variable_19model/_layers/2/adapt_variance/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable0model/_layers/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCallStatefulPartitionedCallsaver_filenameVariable_28Variable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1VariableConst_2**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_8971
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable_28Variable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_9067��
��
�
__inference__traced_save_8971
file_prefix,
"read_disablecopyonread_variable_28:	 .
$read_1_disablecopyonread_variable_27: >
$read_2_disablecopyonread_variable_26: >
$read_3_disablecopyonread_variable_25: 2
$read_4_disablecopyonread_variable_24: 2
$read_5_disablecopyonread_variable_23: >
$read_6_disablecopyonread_variable_22: @>
$read_7_disablecopyonread_variable_21: @2
$read_8_disablecopyonread_variable_20:@2
$read_9_disablecopyonread_variable_19:@9
%read_10_disablecopyonread_variable_18:
�b�9
%read_11_disablecopyonread_variable_17:
�b�4
%read_12_disablecopyonread_variable_16:	�4
%read_13_disablecopyonread_variable_15:	�8
%read_14_disablecopyonread_variable_14:	�8
%read_15_disablecopyonread_variable_13:	�3
%read_16_disablecopyonread_variable_12:3
%read_17_disablecopyonread_variable_11:?
%read_18_disablecopyonread_variable_10: 2
$read_19_disablecopyonread_variable_9: >
$read_20_disablecopyonread_variable_8: @2
$read_21_disablecopyonread_variable_7:@8
$read_22_disablecopyonread_variable_6:
�b�3
$read_23_disablecopyonread_variable_5:	�7
$read_24_disablecopyonread_variable_4:	�2
$read_25_disablecopyonread_variable_3:2
$read_26_disablecopyonread_variable_2:2
$read_27_disablecopyonread_variable_1:,
"read_28_disablecopyonread_variable:	 
savev2_const_2
identity_59��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: e
Read/DisableCopyOnReadDisableCopyOnRead"read_disablecopyonread_variable_28*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp"read_disablecopyonread_variable_28^Read/DisableCopyOnRead*
_output_shapes
: *
dtype0	R
IdentityIdentityRead/ReadVariableOp:value:0*
T0	*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0	*
_output_shapes
: i
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_variable_27*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_variable_27^Read_1/DisableCopyOnRead*
_output_shapes
: *
dtype0V

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_2/DisableCopyOnReadDisableCopyOnRead$read_2_disablecopyonread_variable_26*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp$read_2_disablecopyonread_variable_26^Read_2/DisableCopyOnRead*&
_output_shapes
: *
dtype0f

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*&
_output_shapes
: k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
: i
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_variable_25*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_variable_25^Read_3/DisableCopyOnRead*&
_output_shapes
: *
dtype0f

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*&
_output_shapes
: k

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*&
_output_shapes
: i
Read_4/DisableCopyOnReadDisableCopyOnRead$read_4_disablecopyonread_variable_24*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp$read_4_disablecopyonread_variable_24^Read_4/DisableCopyOnRead*
_output_shapes
: *
dtype0Z

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*
_output_shapes
: _

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_variable_23*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_variable_23^Read_5/DisableCopyOnRead*
_output_shapes
: *
dtype0[
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_6/DisableCopyOnReadDisableCopyOnRead$read_6_disablecopyonread_variable_22*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp$read_6_disablecopyonread_variable_22^Read_6/DisableCopyOnRead*&
_output_shapes
: @*
dtype0g
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*&
_output_shapes
: @m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
: @i
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_variable_21*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_variable_21^Read_7/DisableCopyOnRead*&
_output_shapes
: @*
dtype0g
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*&
_output_shapes
: @m
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*&
_output_shapes
: @i
Read_8/DisableCopyOnReadDisableCopyOnRead$read_8_disablecopyonread_variable_20*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp$read_8_disablecopyonread_variable_20^Read_8/DisableCopyOnRead*
_output_shapes
:@*
dtype0[
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:@i
Read_9/DisableCopyOnReadDisableCopyOnRead$read_9_disablecopyonread_variable_19*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp$read_9_disablecopyonread_variable_19^Read_9/DisableCopyOnRead*
_output_shapes
:@*
dtype0[
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@k
Read_10/DisableCopyOnReadDisableCopyOnRead%read_10_disablecopyonread_variable_18*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp%read_10_disablecopyonread_variable_18^Read_10/DisableCopyOnRead* 
_output_shapes
:
�b�*
dtype0b
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�b�g
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0* 
_output_shapes
:
�b�k
Read_11/DisableCopyOnReadDisableCopyOnRead%read_11_disablecopyonread_variable_17*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp%read_11_disablecopyonread_variable_17^Read_11/DisableCopyOnRead* 
_output_shapes
:
�b�*
dtype0b
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�b�g
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0* 
_output_shapes
:
�b�k
Read_12/DisableCopyOnReadDisableCopyOnRead%read_12_disablecopyonread_variable_16*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp%read_12_disablecopyonread_variable_16^Read_12/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_13/DisableCopyOnReadDisableCopyOnRead%read_13_disablecopyonread_variable_15*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp%read_13_disablecopyonread_variable_15^Read_13/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_14/DisableCopyOnReadDisableCopyOnRead%read_14_disablecopyonread_variable_14*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp%read_14_disablecopyonread_variable_14^Read_14/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:	�k
Read_15/DisableCopyOnReadDisableCopyOnRead%read_15_disablecopyonread_variable_13*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp%read_15_disablecopyonread_variable_13^Read_15/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:	�k
Read_16/DisableCopyOnReadDisableCopyOnRead%read_16_disablecopyonread_variable_12*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp%read_16_disablecopyonread_variable_12^Read_16/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:k
Read_17/DisableCopyOnReadDisableCopyOnRead%read_17_disablecopyonread_variable_11*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp%read_17_disablecopyonread_variable_11^Read_17/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_34IdentityRead_17/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:k
Read_18/DisableCopyOnReadDisableCopyOnRead%read_18_disablecopyonread_variable_10*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp%read_18_disablecopyonread_variable_10^Read_18/DisableCopyOnRead*&
_output_shapes
: *
dtype0h
Identity_36IdentityRead_18/ReadVariableOp:value:0*
T0*&
_output_shapes
: m
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*&
_output_shapes
: j
Read_19/DisableCopyOnReadDisableCopyOnRead$read_19_disablecopyonread_variable_9*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp$read_19_disablecopyonread_variable_9^Read_19/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_38IdentityRead_19/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: j
Read_20/DisableCopyOnReadDisableCopyOnRead$read_20_disablecopyonread_variable_8*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp$read_20_disablecopyonread_variable_8^Read_20/DisableCopyOnRead*&
_output_shapes
: @*
dtype0h
Identity_40IdentityRead_20/ReadVariableOp:value:0*
T0*&
_output_shapes
: @m
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*&
_output_shapes
: @j
Read_21/DisableCopyOnReadDisableCopyOnRead$read_21_disablecopyonread_variable_7*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp$read_21_disablecopyonread_variable_7^Read_21/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_42IdentityRead_21/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_22/DisableCopyOnReadDisableCopyOnRead$read_22_disablecopyonread_variable_6*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp$read_22_disablecopyonread_variable_6^Read_22/DisableCopyOnRead* 
_output_shapes
:
�b�*
dtype0b
Identity_44IdentityRead_22/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�b�g
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0* 
_output_shapes
:
�b�j
Read_23/DisableCopyOnReadDisableCopyOnRead$read_23_disablecopyonread_variable_5*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp$read_23_disablecopyonread_variable_5^Read_23/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_46IdentityRead_23/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_24/DisableCopyOnReadDisableCopyOnRead$read_24_disablecopyonread_variable_4*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp$read_24_disablecopyonread_variable_4^Read_24/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_48IdentityRead_24/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:	�j
Read_25/DisableCopyOnReadDisableCopyOnRead$read_25_disablecopyonread_variable_3*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp$read_25_disablecopyonread_variable_3^Read_25/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_50IdentityRead_25/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:j
Read_26/DisableCopyOnReadDisableCopyOnRead$read_26_disablecopyonread_variable_2*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp$read_26_disablecopyonread_variable_2^Read_26/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_52IdentityRead_26/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:j
Read_27/DisableCopyOnReadDisableCopyOnRead$read_27_disablecopyonread_variable_1*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp$read_27_disablecopyonread_variable_1^Read_27/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_54IdentityRead_27/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:h
Read_28/DisableCopyOnReadDisableCopyOnRead"read_28_disablecopyonread_variable*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp"read_28_disablecopyonread_variable^Read_28/DisableCopyOnRead*
_output_shapes
: *
dtype0	X
Identity_56IdentityRead_28/ReadVariableOp:value:0*
T0	*
_output_shapes
: ]
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0	*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B5model/optimizer/iterations/.ATTRIBUTES/VARIABLE_VALUEB9model/optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEBAmodel/optimizer/_trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEBAmodel/optimizer/_trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEBAmodel/optimizer/_trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEBAmodel/optimizer/_trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEBAmodel/optimizer/_trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEBAmodel/optimizer/_trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEBAmodel/optimizer/_trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEBAmodel/optimizer/_trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB5model/_layers/2/adapt_mean/.ATTRIBUTES/VARIABLE_VALUEB9model/_layers/2/adapt_variance/.ATTRIBUTES/VARIABLE_VALUEB0model/_layers/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0savev2_const_2"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *,
dtypes"
 2		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_58Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_59IdentityIdentity_58:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_59Identity_59:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:?;

_output_shapes
: 
!
_user_specified_name	Const_2:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_15:+'
%
_user_specified_nameVariable_16:+'
%
_user_specified_nameVariable_17:+'
%
_user_specified_nameVariable_18:+
'
%
_user_specified_nameVariable_19:+	'
%
_user_specified_nameVariable_20:+'
%
_user_specified_nameVariable_21:+'
%
_user_specified_nameVariable_22:+'
%
_user_specified_nameVariable_23:+'
%
_user_specified_nameVariable_24:+'
%
_user_specified_nameVariable_25:+'
%
_user_specified_nameVariable_26:+'
%
_user_specified_nameVariable_27:+'
%
_user_specified_nameVariable_28:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�C
�
 __inference_serving_default_8659

inputs&
"functional_1_normalization_1_sub_y'
#functional_1_normalization_1_sqrt_xS
9functional_1_conv2d_1_convolution_readvariableop_resource: C
5functional_1_conv2d_1_reshape_readvariableop_resource: U
;functional_1_conv2d_1_2_convolution_readvariableop_resource: @E
7functional_1_conv2d_1_2_reshape_readvariableop_resource:@E
1functional_1_dense_1_cast_readvariableop_resource:
�b�?
0functional_1_dense_1_add_readvariableop_resource:	�F
3functional_1_dense_1_2_cast_readvariableop_resource:	�@
2functional_1_dense_1_2_add_readvariableop_resource:
identity��,functional_1/conv2d_1/Reshape/ReadVariableOp�0functional_1/conv2d_1/convolution/ReadVariableOp�.functional_1/conv2d_1_2/Reshape/ReadVariableOp�2functional_1/conv2d_1_2/convolution/ReadVariableOp�'functional_1/dense_1/Add/ReadVariableOp�(functional_1/dense_1/Cast/ReadVariableOp�)functional_1/dense_1_2/Add/ReadVariableOp�*functional_1/dense_1_2/Cast/ReadVariableOpt
#functional_1/resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        �
-functional_1/resizing_1/resize/ResizeBilinearResizeBilinearinputs,functional_1/resizing_1/resize/size:output:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(�
 functional_1/normalization_1/SubSub>functional_1/resizing_1/resize/ResizeBilinear:resized_images:0"functional_1_normalization_1_sub_y*
T0*/
_output_shapes
:���������  
!functional_1/normalization_1/SqrtSqrt#functional_1_normalization_1_sqrt_x*
T0*&
_output_shapes
:g
"functional_1/normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
$functional_1/normalization_1/MaximumMaximum%functional_1/normalization_1/Sqrt:y:0+functional_1/normalization_1/Const:output:0*
T0*&
_output_shapes
:�
$functional_1/normalization_1/truedivRealDiv$functional_1/normalization_1/Sub:z:0(functional_1/normalization_1/Maximum:z:0*
T0*/
_output_shapes
:���������  �
0functional_1/conv2d_1/convolution/ReadVariableOpReadVariableOp9functional_1_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
!functional_1/conv2d_1/convolutionConv2D(functional_1/normalization_1/truediv:z:08functional_1/conv2d_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
,functional_1/conv2d_1/Reshape/ReadVariableOpReadVariableOp5functional_1_conv2d_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype0|
#functional_1/conv2d_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
functional_1/conv2d_1/ReshapeReshape4functional_1/conv2d_1/Reshape/ReadVariableOp:value:0,functional_1/conv2d_1/Reshape/shape:output:0*
T0*&
_output_shapes
: �
functional_1/conv2d_1/addAddV2*functional_1/conv2d_1/convolution:output:0&functional_1/conv2d_1/Reshape:output:0*
T0*/
_output_shapes
:��������� {
functional_1/conv2d_1/ReluRelufunctional_1/conv2d_1/add:z:0*
T0*/
_output_shapes
:��������� �
2functional_1/conv2d_1_2/convolution/ReadVariableOpReadVariableOp;functional_1_conv2d_1_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
#functional_1/conv2d_1_2/convolutionConv2D(functional_1/conv2d_1/Relu:activations:0:functional_1/conv2d_1_2/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
.functional_1/conv2d_1_2/Reshape/ReadVariableOpReadVariableOp7functional_1_conv2d_1_2_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0~
%functional_1/conv2d_1_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
functional_1/conv2d_1_2/ReshapeReshape6functional_1/conv2d_1_2/Reshape/ReadVariableOp:value:0.functional_1/conv2d_1_2/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
functional_1/conv2d_1_2/addAddV2,functional_1/conv2d_1_2/convolution:output:0(functional_1/conv2d_1_2/Reshape:output:0*
T0*/
_output_shapes
:���������@
functional_1/conv2d_1_2/ReluRelufunctional_1/conv2d_1_2/add:z:0*
T0*/
_output_shapes
:���������@�
&functional_1/max_pooling2d_1/MaxPool2dMaxPool*functional_1/conv2d_1_2/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
u
$functional_1/flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"���� 1  �
functional_1/flatten_1/ReshapeReshape/functional_1/max_pooling2d_1/MaxPool2d:output:0-functional_1/flatten_1/Reshape/shape:output:0*
T0*(
_output_shapes
:����������b�
(functional_1/dense_1/Cast/ReadVariableOpReadVariableOp1functional_1_dense_1_cast_readvariableop_resource* 
_output_shapes
:
�b�*
dtype0�
functional_1/dense_1/MatMulMatMul'functional_1/flatten_1/Reshape:output:00functional_1/dense_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'functional_1/dense_1/Add/ReadVariableOpReadVariableOp0functional_1_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
functional_1/dense_1/AddAddV2%functional_1/dense_1/MatMul:product:0/functional_1/dense_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������r
functional_1/dense_1/ReluRelufunctional_1/dense_1/Add:z:0*
T0*(
_output_shapes
:�����������
*functional_1/dense_1_2/Cast/ReadVariableOpReadVariableOp3functional_1_dense_1_2_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
functional_1/dense_1_2/MatMulMatMul'functional_1/dense_1/Relu:activations:02functional_1/dense_1_2/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)functional_1/dense_1_2/Add/ReadVariableOpReadVariableOp2functional_1_dense_1_2_add_readvariableop_resource*
_output_shapes
:*
dtype0�
functional_1/dense_1_2/AddAddV2'functional_1/dense_1_2/MatMul:product:01functional_1/dense_1_2/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������m
IdentityIdentityfunctional_1/dense_1_2/Add:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^functional_1/conv2d_1/Reshape/ReadVariableOp1^functional_1/conv2d_1/convolution/ReadVariableOp/^functional_1/conv2d_1_2/Reshape/ReadVariableOp3^functional_1/conv2d_1_2/convolution/ReadVariableOp(^functional_1/dense_1/Add/ReadVariableOp)^functional_1/dense_1/Cast/ReadVariableOp*^functional_1/dense_1_2/Add/ReadVariableOp+^functional_1/dense_1_2/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:���������|�::: : : : : : : : 2\
,functional_1/conv2d_1/Reshape/ReadVariableOp,functional_1/conv2d_1/Reshape/ReadVariableOp2d
0functional_1/conv2d_1/convolution/ReadVariableOp0functional_1/conv2d_1/convolution/ReadVariableOp2`
.functional_1/conv2d_1_2/Reshape/ReadVariableOp.functional_1/conv2d_1_2/Reshape/ReadVariableOp2h
2functional_1/conv2d_1_2/convolution/ReadVariableOp2functional_1/conv2d_1_2/convolution/ReadVariableOp2R
'functional_1/dense_1/Add/ReadVariableOp'functional_1/dense_1/Add/ReadVariableOp2T
(functional_1/dense_1/Cast/ReadVariableOp(functional_1/dense_1/Cast/ReadVariableOp2V
)functional_1/dense_1_2/Add/ReadVariableOp)functional_1/dense_1_2/Add/ReadVariableOp2X
*functional_1/dense_1_2/Cast/ReadVariableOp*functional_1/dense_1_2/Cast/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:,(
&
_output_shapes
::,(
&
_output_shapes
::X T
0
_output_shapes
:���������|�
 
_user_specified_nameinputs
��
�
__inference___call___8565
x&
"sequential_1_normalization_1_sub_y'
#sequential_1_normalization_1_sqrt_xS
9sequential_1_conv2d_1_convolution_readvariableop_resource: C
5sequential_1_conv2d_1_reshape_readvariableop_resource: U
;sequential_1_conv2d_1_2_convolution_readvariableop_resource: @E
7sequential_1_conv2d_1_2_reshape_readvariableop_resource:@E
1sequential_1_dense_1_cast_readvariableop_resource:
�b�?
0sequential_1_dense_1_add_readvariableop_resource:	�F
3sequential_1_dense_1_2_cast_readvariableop_resource:	�@
2sequential_1_dense_1_2_add_readvariableop_resource:
identity	

identity_1

identity_2��,sequential_1/conv2d_1/Reshape/ReadVariableOp�0sequential_1/conv2d_1/convolution/ReadVariableOp�.sequential_1/conv2d_1_2/Reshape/ReadVariableOp�2sequential_1/conv2d_1_2/convolution/ReadVariableOp�'sequential_1/dense_1/Add/ReadVariableOp�(sequential_1/dense_1/Cast/ReadVariableOp�)sequential_1/dense_1_2/Add/ReadVariableOp�*sequential_1/dense_1_2/Cast/ReadVariableOpT
stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :�R
stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :�M

stft/ConstConst*
_output_shapes
: *
dtype0*
value
B :�Z
stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������O
stft/frame/ShapeShapex*
T0*
_output_shapes
::��Q
stft/frame/RankConst*
_output_shapes
: *
dtype0*
value	B :X
stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : X
stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/rangeRangestft/frame/range/start:output:0stft/frame/Rank:output:0stft/frame/range/delta:output:0*
_output_shapes
:q
stft/frame/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 stft/frame/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 stft/frame/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
stft/frame/strided_sliceStridedSlicestft/frame/range:output:0'stft/frame/strided_slice/stack:output:0)stft/frame/strided_slice/stack_1:output:0)stft/frame/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
stft/frame/sub/yConst*
_output_shapes
: *
dtype0*
value	B :k
stft/frame/subSubstft/frame/Rank:output:0stft/frame/sub/y:output:0*
T0*
_output_shapes
: o
stft/frame/sub_1Substft/frame/sub:z:0!stft/frame/strided_slice:output:0*
T0*
_output_shapes
: U
stft/frame/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/packedPack!stft/frame/strided_slice:output:0stft/frame/packed/1:output:0stft/frame/sub_1:z:0*
N*
T0*
_output_shapes
:\
stft/frame/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/splitSplitVstft/frame/Shape:output:0stft/frame/packed:output:0#stft/frame/split/split_dim:output:0*

Tlen0*
T0*$
_output_shapes
::: *
	num_split[
stft/frame/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB ]
stft/frame/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB ~
stft/frame/ReshapeReshapestft/frame/split:output:1#stft/frame/Reshape/shape_1:output:0*
T0*
_output_shapes
: Q
stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :S
stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : R
stft/frame/ConstConst*
_output_shapes
: *
dtype0*
value	B : q
stft/frame/sub_2Substft/frame/Reshape:output:0stft/frame_length:output:0*
T0*
_output_shapes
: p
stft/frame/floordivFloorDivstft/frame/sub_2:z:0stft/frame_step:output:0*
T0*
_output_shapes
: R
stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :l
stft/frame/addAddV2stft/frame/add/x:output:0stft/frame/floordiv:z:0*
T0*
_output_shapes
: m
stft/frame/MaximumMaximumstft/frame/Const:output:0stft/frame/add:z:0*
T0*
_output_shapes
: V
stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :Y
stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/floordiv_1FloorDivstft/frame_length:output:0 stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: Y
stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :~
stft/frame/floordiv_2FloorDivstft/frame_step:output:0 stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: Y
stft/frame/floordiv_3/yConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/floordiv_3FloorDivstft/frame/Reshape:output:0 stft/frame/floordiv_3/y:output:0*
T0*
_output_shapes
: R
stft/frame/mul/yConst*
_output_shapes
: *
dtype0*
value	B :l
stft/frame/mulMulstft/frame/floordiv_3:z:0stft/frame/mul/y:output:0*
T0*
_output_shapes
: d
stft/frame/concat/values_1Packstft/frame/mul:z:0*
N*
T0*
_output_shapes
:X
stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concatConcatV2stft/frame/split:output:0#stft/frame/concat/values_1:output:0stft/frame/split:output:2stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:`
stft/frame/concat_1/values_1/1Const*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/concat_1/values_1Packstft/frame/floordiv_3:z:0'stft/frame/concat_1/values_1/1:output:0*
N*
T0*
_output_shapes
:Z
stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concat_1ConcatV2stft/frame/split:output:0%stft/frame/concat_1/values_1:output:0stft/frame/split:output:2!stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:_
stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: t
*stft/frame/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:\
stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/ones_likeFill3stft/frame/ones_like/Shape/shape_as_tensor:output:0#stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:�
stft/frame/StridedSliceStridedSlicexstft/frame/zeros_like:output:0stft/frame/concat:output:0stft/frame/ones_like:output:0*
Index0*
T0*0
_output_shapes
:�������������������
stft/frame/Reshape_1Reshape stft/frame/StridedSlice:output:0stft/frame/concat_1:output:0*
T0*4
_output_shapes"
 :������������������Z
stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : Z
stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/range_1Range!stft/frame/range_1/start:output:0stft/frame/Maximum:z:0!stft/frame/range_1/delta:output:0*#
_output_shapes
:���������}
stft/frame/mul_1Mulstft/frame/range_1:output:0stft/frame/floordiv_2:z:0*
T0*#
_output_shapes
:���������^
stft/frame/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/Reshape_2/shapePackstft/frame/Maximum:z:0%stft/frame/Reshape_2/shape/1:output:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_2Reshapestft/frame/mul_1:z:0#stft/frame/Reshape_2/shape:output:0*
T0*'
_output_shapes
:���������Z
stft/frame/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : Z
stft/frame/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/range_2Range!stft/frame/range_2/start:output:0stft/frame/floordiv_1:z:0!stft/frame/range_2/delta:output:0*
_output_shapes	
:�^
stft/frame/Reshape_3/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/Reshape_3/shapePack%stft/frame/Reshape_3/shape/0:output:0stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_3Reshapestft/frame/range_2:output:0#stft/frame/Reshape_3/shape:output:0*
T0*
_output_shapes
:	��
stft/frame/add_1AddV2stft/frame/Reshape_2:output:0stft/frame/Reshape_3:output:0*
T0*(
_output_shapes
:����������}
stft/frame/packed_1Packstft/frame/Maximum:z:0stft/frame_length:output:0*
N*
T0*
_output_shapes
:�
stft/frame/GatherV2GatherV2stft/frame/Reshape_1:output:0stft/frame/add_1:z:0!stft/frame/strided_slice:output:0*
Taxis0*
Tindices0*
Tparams0*9
_output_shapes'
%:#�������������������Z
stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concat_2ConcatV2stft/frame/split:output:0stft/frame/packed_1:output:0stft/frame/split:output:2!stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_4Reshapestft/frame/GatherV2:output:0stft/frame/concat_2:output:0*
T0*,
_output_shapes
:���������|�[
stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Zq
stft/hann_window/CastCast"stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: X
stft/hann_window/mod/yConst*
_output_shapes
: *
dtype0*
value	B :~
stft/hann_window/modFloorModstft/frame_length:output:0stft/hann_window/mod/y:output:0*
T0*
_output_shapes
: X
stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :w
stft/hann_window/subSubstft/hann_window/sub/x:output:0stft/hann_window/mod:z:0*
T0*
_output_shapes
: q
stft/hann_window/mulMulstft/hann_window/Cast:y:0stft/hann_window/sub:z:0*
T0*
_output_shapes
: t
stft/hann_window/addAddV2stft/frame_length:output:0stft/hann_window/mul:z:0*
T0*
_output_shapes
: Z
stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
stft/hann_window/sub_1Substft/hann_window/add:z:0!stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: k
stft/hann_window/Cast_1Caststft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ^
stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : ^
stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/hann_window/rangeRange%stft/hann_window/range/start:output:0stft/frame_length:output:0%stft/hann_window/range/delta:output:0*
_output_shapes	
:�u
stft/hann_window/Cast_2Caststft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:�[
stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��@�
stft/hann_window/mul_1Mulstft/hann_window/Const:output:0stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:��
stft/hann_window/truedivRealDivstft/hann_window/mul_1:z:0stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:�_
stft/hann_window/CosCosstft/hann_window/truediv:z:0*
T0*
_output_shapes	
:�]
stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
stft/hann_window/mul_2Mul!stft/hann_window/mul_2/x:output:0stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:�]
stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
stft/hann_window/sub_2Sub!stft/hann_window/sub_2/x:output:0stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:��
stft/mulMulstft/frame/Reshape_4:output:0stft/hann_window/sub_2:z:0*
T0*,
_output_shapes
:���������|�[
stft/rfft/packedPackstft/Const:output:0*
N*
T0*
_output_shapes
:
stft/rfft/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       z
stft/rfft/PadPadstft/mul:z:0stft/rfft/Pad/paddings:output:0*
T0*,
_output_shapes
:���������|�_
stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:�v
	stft/rfftRFFTstft/rfft/Pad:output:0stft/rfft/fft_length:output:0*,
_output_shapes
:���������|�S
Abs
ComplexAbsstft/rfft:output:0*,
_output_shapes
:���������|�d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceAbs:y:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:���������|�*
ellipsis_mask*
new_axis_maskt
#sequential_1/resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        �
-sequential_1/resizing_1/resize/ResizeBilinearResizeBilinearstrided_slice:output:0,sequential_1/resizing_1/resize/size:output:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(�
 sequential_1/normalization_1/SubSub>sequential_1/resizing_1/resize/ResizeBilinear:resized_images:0"sequential_1_normalization_1_sub_y*
T0*/
_output_shapes
:���������  
!sequential_1/normalization_1/SqrtSqrt#sequential_1_normalization_1_sqrt_x*
T0*&
_output_shapes
:g
"sequential_1/normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
$sequential_1/normalization_1/MaximumMaximum%sequential_1/normalization_1/Sqrt:y:0+sequential_1/normalization_1/Const:output:0*
T0*&
_output_shapes
:�
$sequential_1/normalization_1/truedivRealDiv$sequential_1/normalization_1/Sub:z:0(sequential_1/normalization_1/Maximum:z:0*
T0*/
_output_shapes
:���������  �
0sequential_1/conv2d_1/convolution/ReadVariableOpReadVariableOp9sequential_1_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
!sequential_1/conv2d_1/convolutionConv2D(sequential_1/normalization_1/truediv:z:08sequential_1/conv2d_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
,sequential_1/conv2d_1/Reshape/ReadVariableOpReadVariableOp5sequential_1_conv2d_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype0|
#sequential_1/conv2d_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
sequential_1/conv2d_1/ReshapeReshape4sequential_1/conv2d_1/Reshape/ReadVariableOp:value:0,sequential_1/conv2d_1/Reshape/shape:output:0*
T0*&
_output_shapes
: �
sequential_1/conv2d_1/addAddV2*sequential_1/conv2d_1/convolution:output:0&sequential_1/conv2d_1/Reshape:output:0*
T0*/
_output_shapes
:��������� {
sequential_1/conv2d_1/ReluRelusequential_1/conv2d_1/add:z:0*
T0*/
_output_shapes
:��������� �
2sequential_1/conv2d_1_2/convolution/ReadVariableOpReadVariableOp;sequential_1_conv2d_1_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
#sequential_1/conv2d_1_2/convolutionConv2D(sequential_1/conv2d_1/Relu:activations:0:sequential_1/conv2d_1_2/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
.sequential_1/conv2d_1_2/Reshape/ReadVariableOpReadVariableOp7sequential_1_conv2d_1_2_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0~
%sequential_1/conv2d_1_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
sequential_1/conv2d_1_2/ReshapeReshape6sequential_1/conv2d_1_2/Reshape/ReadVariableOp:value:0.sequential_1/conv2d_1_2/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
sequential_1/conv2d_1_2/addAddV2,sequential_1/conv2d_1_2/convolution:output:0(sequential_1/conv2d_1_2/Reshape:output:0*
T0*/
_output_shapes
:���������@
sequential_1/conv2d_1_2/ReluRelusequential_1/conv2d_1_2/add:z:0*
T0*/
_output_shapes
:���������@�
&sequential_1/max_pooling2d_1/MaxPool2dMaxPool*sequential_1/conv2d_1_2/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
u
$sequential_1/flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"���� 1  �
sequential_1/flatten_1/ReshapeReshape/sequential_1/max_pooling2d_1/MaxPool2d:output:0-sequential_1/flatten_1/Reshape/shape:output:0*
T0*(
_output_shapes
:����������b�
(sequential_1/dense_1/Cast/ReadVariableOpReadVariableOp1sequential_1_dense_1_cast_readvariableop_resource* 
_output_shapes
:
�b�*
dtype0�
sequential_1/dense_1/MatMulMatMul'sequential_1/flatten_1/Reshape:output:00sequential_1/dense_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'sequential_1/dense_1/Add/ReadVariableOpReadVariableOp0sequential_1_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_1/dense_1/AddAddV2%sequential_1/dense_1/MatMul:product:0/sequential_1/dense_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������r
sequential_1/dense_1/ReluRelusequential_1/dense_1/Add:z:0*
T0*(
_output_shapes
:�����������
*sequential_1/dense_1_2/Cast/ReadVariableOpReadVariableOp3sequential_1_dense_1_2_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_1/dense_1_2/MatMulMatMul'sequential_1/dense_1/Relu:activations:02sequential_1/dense_1_2/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)sequential_1/dense_1_2/Add/ReadVariableOpReadVariableOp2sequential_1_dense_1_2_add_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/dense_1_2/AddAddV2'sequential_1/dense_1_2/MatMul:product:01sequential_1/dense_1_2/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������y
ArgMaxArgMaxsequential_1/dense_1_2/Add:z:0ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������t
GatherV2/paramsConst*
_output_shapes
:*
dtype0*1
value(B&B_background_noise_BgoBstopO
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
GatherV2GatherV2GatherV2/params:output:0ArgMax:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:���������Z
IdentityIdentityArgMax:output:0^NoOp*
T0	*#
_output_shapes
:���������^

Identity_1IdentityGatherV2:output:0^NoOp*
T0*#
_output_shapes
:���������o

Identity_2Identitysequential_1/dense_1_2/Add:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^sequential_1/conv2d_1/Reshape/ReadVariableOp1^sequential_1/conv2d_1/convolution/ReadVariableOp/^sequential_1/conv2d_1_2/Reshape/ReadVariableOp3^sequential_1/conv2d_1_2/convolution/ReadVariableOp(^sequential_1/dense_1/Add/ReadVariableOp)^sequential_1/dense_1/Cast/ReadVariableOp*^sequential_1/dense_1_2/Add/ReadVariableOp+^sequential_1/dense_1_2/Cast/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������}::: : : : : : : : 2\
,sequential_1/conv2d_1/Reshape/ReadVariableOp,sequential_1/conv2d_1/Reshape/ReadVariableOp2d
0sequential_1/conv2d_1/convolution/ReadVariableOp0sequential_1/conv2d_1/convolution/ReadVariableOp2`
.sequential_1/conv2d_1_2/Reshape/ReadVariableOp.sequential_1/conv2d_1_2/Reshape/ReadVariableOp2h
2sequential_1/conv2d_1_2/convolution/ReadVariableOp2sequential_1/conv2d_1_2/convolution/ReadVariableOp2R
'sequential_1/dense_1/Add/ReadVariableOp'sequential_1/dense_1/Add/ReadVariableOp2T
(sequential_1/dense_1/Cast/ReadVariableOp(sequential_1/dense_1/Cast/ReadVariableOp2V
)sequential_1/dense_1_2/Add/ReadVariableOp)sequential_1/dense_1_2/Add/ReadVariableOp2X
*sequential_1/dense_1_2/Cast/ReadVariableOp*sequential_1/dense_1_2/Cast/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:,(
&
_output_shapes
::,(
&
_output_shapes
::K G
(
_output_shapes
:����������}

_user_specified_namex
�C
�
 __inference_serving_default_8612

inputs&
"sequential_1_normalization_1_sub_y'
#sequential_1_normalization_1_sqrt_xS
9sequential_1_conv2d_1_convolution_readvariableop_resource: C
5sequential_1_conv2d_1_reshape_readvariableop_resource: U
;sequential_1_conv2d_1_2_convolution_readvariableop_resource: @E
7sequential_1_conv2d_1_2_reshape_readvariableop_resource:@E
1sequential_1_dense_1_cast_readvariableop_resource:
�b�?
0sequential_1_dense_1_add_readvariableop_resource:	�F
3sequential_1_dense_1_2_cast_readvariableop_resource:	�@
2sequential_1_dense_1_2_add_readvariableop_resource:
identity��,sequential_1/conv2d_1/Reshape/ReadVariableOp�0sequential_1/conv2d_1/convolution/ReadVariableOp�.sequential_1/conv2d_1_2/Reshape/ReadVariableOp�2sequential_1/conv2d_1_2/convolution/ReadVariableOp�'sequential_1/dense_1/Add/ReadVariableOp�(sequential_1/dense_1/Cast/ReadVariableOp�)sequential_1/dense_1_2/Add/ReadVariableOp�*sequential_1/dense_1_2/Cast/ReadVariableOpt
#sequential_1/resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        �
-sequential_1/resizing_1/resize/ResizeBilinearResizeBilinearinputs,sequential_1/resizing_1/resize/size:output:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(�
 sequential_1/normalization_1/SubSub>sequential_1/resizing_1/resize/ResizeBilinear:resized_images:0"sequential_1_normalization_1_sub_y*
T0*/
_output_shapes
:���������  
!sequential_1/normalization_1/SqrtSqrt#sequential_1_normalization_1_sqrt_x*
T0*&
_output_shapes
:g
"sequential_1/normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
$sequential_1/normalization_1/MaximumMaximum%sequential_1/normalization_1/Sqrt:y:0+sequential_1/normalization_1/Const:output:0*
T0*&
_output_shapes
:�
$sequential_1/normalization_1/truedivRealDiv$sequential_1/normalization_1/Sub:z:0(sequential_1/normalization_1/Maximum:z:0*
T0*/
_output_shapes
:���������  �
0sequential_1/conv2d_1/convolution/ReadVariableOpReadVariableOp9sequential_1_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
!sequential_1/conv2d_1/convolutionConv2D(sequential_1/normalization_1/truediv:z:08sequential_1/conv2d_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
,sequential_1/conv2d_1/Reshape/ReadVariableOpReadVariableOp5sequential_1_conv2d_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype0|
#sequential_1/conv2d_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
sequential_1/conv2d_1/ReshapeReshape4sequential_1/conv2d_1/Reshape/ReadVariableOp:value:0,sequential_1/conv2d_1/Reshape/shape:output:0*
T0*&
_output_shapes
: �
sequential_1/conv2d_1/addAddV2*sequential_1/conv2d_1/convolution:output:0&sequential_1/conv2d_1/Reshape:output:0*
T0*/
_output_shapes
:��������� {
sequential_1/conv2d_1/ReluRelusequential_1/conv2d_1/add:z:0*
T0*/
_output_shapes
:��������� �
2sequential_1/conv2d_1_2/convolution/ReadVariableOpReadVariableOp;sequential_1_conv2d_1_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
#sequential_1/conv2d_1_2/convolutionConv2D(sequential_1/conv2d_1/Relu:activations:0:sequential_1/conv2d_1_2/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
�
.sequential_1/conv2d_1_2/Reshape/ReadVariableOpReadVariableOp7sequential_1_conv2d_1_2_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0~
%sequential_1/conv2d_1_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
sequential_1/conv2d_1_2/ReshapeReshape6sequential_1/conv2d_1_2/Reshape/ReadVariableOp:value:0.sequential_1/conv2d_1_2/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
sequential_1/conv2d_1_2/addAddV2,sequential_1/conv2d_1_2/convolution:output:0(sequential_1/conv2d_1_2/Reshape:output:0*
T0*/
_output_shapes
:���������@
sequential_1/conv2d_1_2/ReluRelusequential_1/conv2d_1_2/add:z:0*
T0*/
_output_shapes
:���������@�
&sequential_1/max_pooling2d_1/MaxPool2dMaxPool*sequential_1/conv2d_1_2/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
u
$sequential_1/flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"���� 1  �
sequential_1/flatten_1/ReshapeReshape/sequential_1/max_pooling2d_1/MaxPool2d:output:0-sequential_1/flatten_1/Reshape/shape:output:0*
T0*(
_output_shapes
:����������b�
(sequential_1/dense_1/Cast/ReadVariableOpReadVariableOp1sequential_1_dense_1_cast_readvariableop_resource* 
_output_shapes
:
�b�*
dtype0�
sequential_1/dense_1/MatMulMatMul'sequential_1/flatten_1/Reshape:output:00sequential_1/dense_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'sequential_1/dense_1/Add/ReadVariableOpReadVariableOp0sequential_1_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_1/dense_1/AddAddV2%sequential_1/dense_1/MatMul:product:0/sequential_1/dense_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������r
sequential_1/dense_1/ReluRelusequential_1/dense_1/Add:z:0*
T0*(
_output_shapes
:�����������
*sequential_1/dense_1_2/Cast/ReadVariableOpReadVariableOp3sequential_1_dense_1_2_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_1/dense_1_2/MatMulMatMul'sequential_1/dense_1/Relu:activations:02sequential_1/dense_1_2/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)sequential_1/dense_1_2/Add/ReadVariableOpReadVariableOp2sequential_1_dense_1_2_add_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/dense_1_2/AddAddV2'sequential_1/dense_1_2/MatMul:product:01sequential_1/dense_1_2/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������m
IdentityIdentitysequential_1/dense_1_2/Add:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^sequential_1/conv2d_1/Reshape/ReadVariableOp1^sequential_1/conv2d_1/convolution/ReadVariableOp/^sequential_1/conv2d_1_2/Reshape/ReadVariableOp3^sequential_1/conv2d_1_2/convolution/ReadVariableOp(^sequential_1/dense_1/Add/ReadVariableOp)^sequential_1/dense_1/Cast/ReadVariableOp*^sequential_1/dense_1_2/Add/ReadVariableOp+^sequential_1/dense_1_2/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:���������|�::: : : : : : : : 2\
,sequential_1/conv2d_1/Reshape/ReadVariableOp,sequential_1/conv2d_1/Reshape/ReadVariableOp2d
0sequential_1/conv2d_1/convolution/ReadVariableOp0sequential_1/conv2d_1/convolution/ReadVariableOp2`
.sequential_1/conv2d_1_2/Reshape/ReadVariableOp.sequential_1/conv2d_1_2/Reshape/ReadVariableOp2h
2sequential_1/conv2d_1_2/convolution/ReadVariableOp2sequential_1/conv2d_1_2/convolution/ReadVariableOp2R
'sequential_1/dense_1/Add/ReadVariableOp'sequential_1/dense_1/Add/ReadVariableOp2T
(sequential_1/dense_1/Cast/ReadVariableOp(sequential_1/dense_1/Cast/ReadVariableOp2V
)sequential_1/dense_1_2/Add/ReadVariableOp)sequential_1/dense_1_2/Add/ReadVariableOp2X
*sequential_1/dense_1_2/Cast/ReadVariableOp*sequential_1/dense_1_2/Cast/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:,(
&
_output_shapes
::,(
&
_output_shapes
::X T
0
_output_shapes
:���������|�
 
_user_specified_nameinputs
��
�
__inference___call___8403
x&
"sequential_1_normalization_1_sub_y'
#sequential_1_normalization_1_sqrt_xS
9sequential_1_conv2d_1_convolution_readvariableop_resource: C
5sequential_1_conv2d_1_reshape_readvariableop_resource: U
;sequential_1_conv2d_1_2_convolution_readvariableop_resource: @E
7sequential_1_conv2d_1_2_reshape_readvariableop_resource:@E
1sequential_1_dense_1_cast_readvariableop_resource:
�b�?
0sequential_1_dense_1_add_readvariableop_resource:	�F
3sequential_1_dense_1_2_cast_readvariableop_resource:	�@
2sequential_1_dense_1_2_add_readvariableop_resource:
identity	

identity_1

identity_2��,sequential_1/conv2d_1/Reshape/ReadVariableOp�0sequential_1/conv2d_1/convolution/ReadVariableOp�.sequential_1/conv2d_1_2/Reshape/ReadVariableOp�2sequential_1/conv2d_1_2/convolution/ReadVariableOp�'sequential_1/dense_1/Add/ReadVariableOp�(sequential_1/dense_1/Cast/ReadVariableOp�)sequential_1/dense_1_2/Add/ReadVariableOp�*sequential_1/dense_1_2/Cast/ReadVariableOp/
ReadFileReadFilex*
_output_shapes
: ~
	DecodeWav	DecodeWavReadFile:contents:0*!
_output_shapes
:	�}: *
desired_channels*
desired_samples�}k
SqueezeSqueezeDecodeWav:audio:0*
T0*
_output_shapes	
:�}*
squeeze_dims

���������d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceSqueeze:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�}*

begin_mask*
end_mask*
new_axis_maskT
stft/frame_lengthConst*
_output_shapes
: *
dtype0*
value
B :�R
stft/frame_stepConst*
_output_shapes
: *
dtype0*
value
B :�M

stft/ConstConst*
_output_shapes
: *
dtype0*
value
B :�Z
stft/frame/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������a
stft/frame/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   �>  _
stft/frame/Size/ConstConst*
_output_shapes
:*
dtype0*
valueB:Q
stft/frame/SizeConst*
_output_shapes
: *
dtype0*
value	B :Z
stft/frame/Size_1/ConstConst*
_output_shapes
: *
dtype0*
valueB S
stft/frame/Size_1Const*
_output_shapes
: *
dtype0*
value	B : R
stft/frame/ConstConst*
_output_shapes
: *
dtype0*
value	B : S
stft/frame/sub/xConst*
_output_shapes
: *
dtype0*
value
B :�}m
stft/frame/subSubstft/frame/sub/x:output:0stft/frame_length:output:0*
T0*
_output_shapes
: n
stft/frame/floordivFloorDivstft/frame/sub:z:0stft/frame_step:output:0*
T0*
_output_shapes
: R
stft/frame/add/xConst*
_output_shapes
: *
dtype0*
value	B :l
stft/frame/addAddV2stft/frame/add/x:output:0stft/frame/floordiv:z:0*
T0*
_output_shapes
: m
stft/frame/MaximumMaximumstft/frame/Const:output:0stft/frame/add:z:0*
T0*
_output_shapes
: V
stft/frame/gcd/ConstConst*
_output_shapes
: *
dtype0*
value	B :Y
stft/frame/floordiv_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/floordiv_1FloorDivstft/frame_length:output:0 stft/frame/floordiv_1/y:output:0*
T0*
_output_shapes
: Y
stft/frame/floordiv_2/yConst*
_output_shapes
: *
dtype0*
value	B :~
stft/frame/floordiv_2FloorDivstft/frame_step:output:0 stft/frame/floordiv_2/y:output:0*
T0*
_output_shapes
: d
stft/frame/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:e
stft/frame/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:�}]
stft/frame/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB X
stft/frame/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concatConcatV2#stft/frame/concat/values_0:output:0#stft/frame/concat/values_1:output:0#stft/frame/concat/values_2:output:0stft/frame/concat/axis:output:0*
N*
T0*
_output_shapes
:f
stft/frame/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:m
stft/frame/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB"�>     _
stft/frame/concat_1/values_2Const*
_output_shapes
: *
dtype0*
valueB Z
stft/frame/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concat_1ConcatV2%stft/frame/concat_1/values_0:output:0%stft/frame/concat_1/values_1:output:0%stft/frame/concat_1/values_2:output:0!stft/frame/concat_1/axis:output:0*
N*
T0*
_output_shapes
:m
stft/frame/zeros_like/tensorConst*
_output_shapes
:*
dtype0*
valueB"   �>  _
stft/frame/zeros_likeConst*
_output_shapes
:*
dtype0*
valueB: t
*stft/frame/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:\
stft/frame/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/ones_likeFill3stft/frame/ones_like/Shape/shape_as_tensor:output:0#stft/frame/ones_like/Const:output:0*
T0*
_output_shapes
:�
stft/frame/StridedSliceStridedSlicestrided_slice:output:0stft/frame/zeros_like:output:0stft/frame/concat:output:0stft/frame/ones_like:output:0*
Index0*
T0*
_output_shapes
:	�}�
stft/frame/ReshapeReshape stft/frame/StridedSlice:output:0stft/frame/concat_1:output:0*
T0*#
_output_shapes
:�}X
stft/frame/range/startConst*
_output_shapes
: *
dtype0*
value	B : X
stft/frame/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/rangeRangestft/frame/range/start:output:0stft/frame/Maximum:z:0stft/frame/range/delta:output:0*
_output_shapes
:|p
stft/frame/mulMulstft/frame/range:output:0stft/frame/floordiv_2:z:0*
T0*
_output_shapes
:|^
stft/frame/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/Reshape_1/shapePackstft/frame/Maximum:z:0%stft/frame/Reshape_1/shape/1:output:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_1Reshapestft/frame/mul:z:0#stft/frame/Reshape_1/shape:output:0*
T0*
_output_shapes

:|Z
stft/frame/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : Z
stft/frame/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/range_1Range!stft/frame/range_1/start:output:0stft/frame/floordiv_1:z:0!stft/frame/range_1/delta:output:0*
_output_shapes	
:�^
stft/frame/Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/Reshape_2/shapePack%stft/frame/Reshape_2/shape/0:output:0stft/frame/floordiv_1:z:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_2Reshapestft/frame/range_1:output:0#stft/frame/Reshape_2/shape:output:0*
T0*
_output_shapes
:	��
stft/frame/add_1AddV2stft/frame/Reshape_1:output:0stft/frame/Reshape_2:output:0*
T0*
_output_shapes
:	|�\
stft/frame/Const_1Const*
_output_shapes
:*
dtype0*
valueB:U
stft/frame/Const_2Const*
_output_shapes
: *
dtype0*
valueB {
stft/frame/packedPackstft/frame/Maximum:z:0stft/frame_length:output:0*
N*
T0*
_output_shapes
:Z
stft/frame/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :�
stft/frame/GatherV2GatherV2stft/frame/Reshape:output:0stft/frame/add_1:z:0!stft/frame/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:|�Z
stft/frame/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
stft/frame/concat_2ConcatV2stft/frame/Const_1:output:0stft/frame/packed:output:0stft/frame/Const_2:output:0!stft/frame/concat_2/axis:output:0*
N*
T0*
_output_shapes
:�
stft/frame/Reshape_3Reshapestft/frame/GatherV2:output:0stft/frame/concat_2:output:0*
T0*#
_output_shapes
:|�[
stft/hann_window/periodicConst*
_output_shapes
: *
dtype0
*
value	B
 Zq
stft/hann_window/CastCast"stft/hann_window/periodic:output:0*

DstT0*

SrcT0
*
_output_shapes
: X
stft/hann_window/mod/yConst*
_output_shapes
: *
dtype0*
value	B :~
stft/hann_window/modFloorModstft/frame_length:output:0stft/hann_window/mod/y:output:0*
T0*
_output_shapes
: X
stft/hann_window/sub/xConst*
_output_shapes
: *
dtype0*
value	B :w
stft/hann_window/subSubstft/hann_window/sub/x:output:0stft/hann_window/mod:z:0*
T0*
_output_shapes
: q
stft/hann_window/mulMulstft/hann_window/Cast:y:0stft/hann_window/sub:z:0*
T0*
_output_shapes
: t
stft/hann_window/addAddV2stft/frame_length:output:0stft/hann_window/mul:z:0*
T0*
_output_shapes
: Z
stft/hann_window/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
stft/hann_window/sub_1Substft/hann_window/add:z:0!stft/hann_window/sub_1/y:output:0*
T0*
_output_shapes
: k
stft/hann_window/Cast_1Caststft/hann_window/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ^
stft/hann_window/range/startConst*
_output_shapes
: *
dtype0*
value	B : ^
stft/hann_window/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
stft/hann_window/rangeRange%stft/hann_window/range/start:output:0stft/frame_length:output:0%stft/hann_window/range/delta:output:0*
_output_shapes	
:�u
stft/hann_window/Cast_2Caststft/hann_window/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:�[
stft/hann_window/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *��@�
stft/hann_window/mul_1Mulstft/hann_window/Const:output:0stft/hann_window/Cast_2:y:0*
T0*
_output_shapes	
:��
stft/hann_window/truedivRealDivstft/hann_window/mul_1:z:0stft/hann_window/Cast_1:y:0*
T0*
_output_shapes	
:�_
stft/hann_window/CosCosstft/hann_window/truediv:z:0*
T0*
_output_shapes	
:�]
stft/hann_window/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
stft/hann_window/mul_2Mul!stft/hann_window/mul_2/x:output:0stft/hann_window/Cos:y:0*
T0*
_output_shapes	
:�]
stft/hann_window/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
stft/hann_window/sub_2Sub!stft/hann_window/sub_2/x:output:0stft/hann_window/mul_2:z:0*
T0*
_output_shapes	
:�x
stft/mulMulstft/frame/Reshape_3:output:0stft/hann_window/sub_2:z:0*
T0*#
_output_shapes
:|�[
stft/rfft/packedPackstft/Const:output:0*
N*
T0*
_output_shapes
:
stft/rfft/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"                       q
stft/rfft/PadPadstft/mul:z:0stft/rfft/Pad/paddings:output:0*
T0*#
_output_shapes
:|�_
stft/rfft/fft_lengthConst*
_output_shapes
:*
dtype0*
valueB:�m
	stft/rfftRFFTstft/rfft/Pad:output:0stft/rfft/fft_length:output:0*#
_output_shapes
:|�J
Abs
ComplexAbsstft/rfft:output:0*#
_output_shapes
:|�f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceAbs:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:|�*
ellipsis_mask*
new_axis_maskt
#sequential_1/resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"        �
-sequential_1/resizing_1/resize/ResizeBilinearResizeBilinearstrided_slice_1:output:0,sequential_1/resizing_1/resize/size:output:0*
T0*&
_output_shapes
:  *
half_pixel_centers(�
 sequential_1/normalization_1/SubSub>sequential_1/resizing_1/resize/ResizeBilinear:resized_images:0"sequential_1_normalization_1_sub_y*
T0*&
_output_shapes
:  
!sequential_1/normalization_1/SqrtSqrt#sequential_1_normalization_1_sqrt_x*
T0*&
_output_shapes
:g
"sequential_1/normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
$sequential_1/normalization_1/MaximumMaximum%sequential_1/normalization_1/Sqrt:y:0+sequential_1/normalization_1/Const:output:0*
T0*&
_output_shapes
:�
$sequential_1/normalization_1/truedivRealDiv$sequential_1/normalization_1/Sub:z:0(sequential_1/normalization_1/Maximum:z:0*
T0*&
_output_shapes
:  �
0sequential_1/conv2d_1/convolution/ReadVariableOpReadVariableOp9sequential_1_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
!sequential_1/conv2d_1/convolutionConv2D(sequential_1/normalization_1/truediv:z:08sequential_1/conv2d_1/convolution/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
�
,sequential_1/conv2d_1/Reshape/ReadVariableOpReadVariableOp5sequential_1_conv2d_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype0|
#sequential_1/conv2d_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
sequential_1/conv2d_1/ReshapeReshape4sequential_1/conv2d_1/Reshape/ReadVariableOp:value:0,sequential_1/conv2d_1/Reshape/shape:output:0*
T0*&
_output_shapes
: �
sequential_1/conv2d_1/addAddV2*sequential_1/conv2d_1/convolution:output:0&sequential_1/conv2d_1/Reshape:output:0*
T0*&
_output_shapes
: r
sequential_1/conv2d_1/ReluRelusequential_1/conv2d_1/add:z:0*
T0*&
_output_shapes
: �
2sequential_1/conv2d_1_2/convolution/ReadVariableOpReadVariableOp;sequential_1_conv2d_1_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
#sequential_1/conv2d_1_2/convolutionConv2D(sequential_1/conv2d_1/Relu:activations:0:sequential_1/conv2d_1_2/convolution/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
�
.sequential_1/conv2d_1_2/Reshape/ReadVariableOpReadVariableOp7sequential_1_conv2d_1_2_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0~
%sequential_1/conv2d_1_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
sequential_1/conv2d_1_2/ReshapeReshape6sequential_1/conv2d_1_2/Reshape/ReadVariableOp:value:0.sequential_1/conv2d_1_2/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
sequential_1/conv2d_1_2/addAddV2,sequential_1/conv2d_1_2/convolution:output:0(sequential_1/conv2d_1_2/Reshape:output:0*
T0*&
_output_shapes
:@v
sequential_1/conv2d_1_2/ReluRelusequential_1/conv2d_1_2/add:z:0*
T0*&
_output_shapes
:@�
&sequential_1/max_pooling2d_1/MaxPool2dMaxPool*sequential_1/conv2d_1_2/Relu:activations:0*&
_output_shapes
:@*
ksize
*
paddingVALID*
strides
u
$sequential_1/flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"    1  �
sequential_1/flatten_1/ReshapeReshape/sequential_1/max_pooling2d_1/MaxPool2d:output:0-sequential_1/flatten_1/Reshape/shape:output:0*
T0*
_output_shapes
:	�b�
(sequential_1/dense_1/Cast/ReadVariableOpReadVariableOp1sequential_1_dense_1_cast_readvariableop_resource* 
_output_shapes
:
�b�*
dtype0�
sequential_1/dense_1/MatMulMatMul'sequential_1/flatten_1/Reshape:output:00sequential_1/dense_1/Cast/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
'sequential_1/dense_1/Add/ReadVariableOpReadVariableOp0sequential_1_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_1/dense_1/AddAddV2%sequential_1/dense_1/MatMul:product:0/sequential_1/dense_1/Add/ReadVariableOp:value:0*
T0*
_output_shapes
:	�i
sequential_1/dense_1/ReluRelusequential_1/dense_1/Add:z:0*
T0*
_output_shapes
:	��
*sequential_1/dense_1_2/Cast/ReadVariableOpReadVariableOp3sequential_1_dense_1_2_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_1/dense_1_2/MatMulMatMul'sequential_1/dense_1/Relu:activations:02sequential_1/dense_1_2/Cast/ReadVariableOp:value:0*
T0*
_output_shapes

:�
)sequential_1/dense_1_2/Add/ReadVariableOpReadVariableOp2sequential_1_dense_1_2_add_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/dense_1_2/AddAddV2'sequential_1/dense_1_2/MatMul:product:01sequential_1/dense_1_2/Add/ReadVariableOp:value:0*
T0*
_output_shapes

:[
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������p
ArgMaxArgMaxsequential_1/dense_1_2/Add:z:0ArgMax/dimension:output:0*
T0*
_output_shapes
:t
GatherV2/paramsConst*
_output_shapes
:*
dtype0*1
value(B&B_background_noise_BgoBstopO
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
GatherV2GatherV2GatherV2/params:output:0ArgMax:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
IdentityIdentityArgMax:output:0^NoOp*
T0	*
_output_shapes
:U

Identity_1IdentityGatherV2:output:0^NoOp*
T0*
_output_shapes
:f

Identity_2Identitysequential_1/dense_1_2/Add:z:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp-^sequential_1/conv2d_1/Reshape/ReadVariableOp1^sequential_1/conv2d_1/convolution/ReadVariableOp/^sequential_1/conv2d_1_2/Reshape/ReadVariableOp3^sequential_1/conv2d_1_2/convolution/ReadVariableOp(^sequential_1/dense_1/Add/ReadVariableOp)^sequential_1/dense_1/Cast/ReadVariableOp*^sequential_1/dense_1_2/Add/ReadVariableOp+^sequential_1/dense_1_2/Cast/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: ::: : : : : : : : 2\
,sequential_1/conv2d_1/Reshape/ReadVariableOp,sequential_1/conv2d_1/Reshape/ReadVariableOp2d
0sequential_1/conv2d_1/convolution/ReadVariableOp0sequential_1/conv2d_1/convolution/ReadVariableOp2`
.sequential_1/conv2d_1_2/Reshape/ReadVariableOp.sequential_1/conv2d_1_2/Reshape/ReadVariableOp2h
2sequential_1/conv2d_1_2/convolution/ReadVariableOp2sequential_1/conv2d_1_2/convolution/ReadVariableOp2R
'sequential_1/dense_1/Add/ReadVariableOp'sequential_1/dense_1/Add/ReadVariableOp2T
(sequential_1/dense_1/Cast/ReadVariableOp(sequential_1/dense_1/Cast/ReadVariableOp2V
)sequential_1/dense_1_2/Add/ReadVariableOp)sequential_1/dense_1_2/Add/ReadVariableOp2X
*sequential_1/dense_1_2/Cast/ReadVariableOp*sequential_1/dense_1_2/Cast/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:,(
&
_output_shapes
::,(
&
_output_shapes
::9 5

_output_shapes
: 

_user_specified_namex
�
�
 __inference__traced_restore_9067
file_prefix&
assignvariableop_variable_28:	 (
assignvariableop_1_variable_27: 8
assignvariableop_2_variable_26: 8
assignvariableop_3_variable_25: ,
assignvariableop_4_variable_24: ,
assignvariableop_5_variable_23: 8
assignvariableop_6_variable_22: @8
assignvariableop_7_variable_21: @,
assignvariableop_8_variable_20:@,
assignvariableop_9_variable_19:@3
assignvariableop_10_variable_18:
�b�3
assignvariableop_11_variable_17:
�b�.
assignvariableop_12_variable_16:	�.
assignvariableop_13_variable_15:	�2
assignvariableop_14_variable_14:	�2
assignvariableop_15_variable_13:	�-
assignvariableop_16_variable_12:-
assignvariableop_17_variable_11:9
assignvariableop_18_variable_10: ,
assignvariableop_19_variable_9: 8
assignvariableop_20_variable_8: @,
assignvariableop_21_variable_7:@2
assignvariableop_22_variable_6:
�b�-
assignvariableop_23_variable_5:	�1
assignvariableop_24_variable_4:	�,
assignvariableop_25_variable_3:,
assignvariableop_26_variable_2:,
assignvariableop_27_variable_1:&
assignvariableop_28_variable:	 
identity_30��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B5model/optimizer/iterations/.ATTRIBUTES/VARIABLE_VALUEB9model/optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB7model/optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB8model/optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEBAmodel/optimizer/_trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEBAmodel/optimizer/_trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEBAmodel/optimizer/_trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEBAmodel/optimizer/_trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEBAmodel/optimizer/_trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEBAmodel/optimizer/_trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEBAmodel/optimizer/_trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEBAmodel/optimizer/_trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB5model/_layers/2/adapt_mean/.ATTRIBUTES/VARIABLE_VALUEB9model/_layers/2/adapt_variance/.ATTRIBUTES/VARIABLE_VALUEB0model/_layers/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_28Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_27Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_26Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_25Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_24Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_23Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_22Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_21Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_20Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_19Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_18Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_17Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_16Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_variable_15Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_14Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_variable_13Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_variable_12Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_variable_11Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_variable_10Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_variable_9Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_variable_8Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_variable_7Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_variable_6Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_variable_5Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_variable_4Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_variable_3Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_variable_2Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_variable_1Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_variableIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_30IdentityIdentity_29:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_30Identity_30:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_15:+'
%
_user_specified_nameVariable_16:+'
%
_user_specified_nameVariable_17:+'
%
_user_specified_nameVariable_18:+
'
%
_user_specified_nameVariable_19:+	'
%
_user_specified_nameVariable_20:+'
%
_user_specified_nameVariable_21:+'
%
_user_specified_nameVariable_22:+'
%
_user_specified_nameVariable_23:+'
%
_user_specified_nameVariable_24:+'
%
_user_specified_nameVariable_25:+'
%
_user_specified_nameVariable_26:+'
%
_user_specified_nameVariable_27:+'
%
_user_specified_nameVariable_28:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:�M
G
	model
__call__

signatures"
_generic_user_object
�
_functional
	optimizer
_default_save_signature
_inbound_nodes
_outbound_nodes
	_losses

	_loss_ids
_losses_override
_layers
_build_shapes_dict"
_generic_user_object
�
trace_0
trace_12�
__inference___call___8403
__inference___call___8565�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
"
signature_map
�
_tracked
_inbound_nodes
_outbound_nodes
_losses
_losses_override
_operations
_layers
_build_shapes_dict
output_names
_default_save_signature"
_generic_user_object
�

_variables
_trainable_variables
 _trainable_variables_indices

iterations
_learning_rate

_momentums
 _velocities"
_generic_user_object
�
!trace_02�
 __inference_serving_default_8612�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!����������|�z!trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
n
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10"
trackable_list_wrapper
 "
trackable_dict_wrapper
�
-	capture_0
.	capture_1B�
__inference___call___8403x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z-	capture_0z.	capture_1
�
-	capture_0
.	capture_1B�
__inference___call___8565x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z-	capture_0z.	capture_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
n
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10"
trackable_list_wrapper
n
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�
/trace_02�
 __inference_serving_default_8659�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!����������|�z/trace_0
�
0
1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15
>16
?17"
trackable_list_wrapper
X
@0
A1
B2
C3
D4
E5
F6
G7"
trackable_list_wrapper
 "
trackable_dict_wrapper
:	 2adam/iteration
: 2adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
-	capture_0
.	capture_1B�
 __inference_serving_default_8612inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z-	capture_0z.	capture_1
y
H_inbound_nodes
I_outbound_nodes
J_losses
K	_loss_ids
L_losses_override"
_generic_user_object
�
M_inbound_nodes
N_outbound_nodes
O_losses
P	_loss_ids
Q_losses_override
R_build_shapes_dict"
_generic_user_object
�
S
adapt_mean
Tadapt_variance
	Ucount
V_inbound_nodes
W_outbound_nodes
X_losses
Y	_loss_ids
Z_losses_override
[_reduce_axis_mask
\_broadcast_shape
]_build_shapes_dict"
_generic_user_object
�
@_kernel
Abias
^_inbound_nodes
__outbound_nodes
`_losses
a	_loss_ids
b_losses_override
c_build_shapes_dict"
_generic_user_object
�
B_kernel
Cbias
d_inbound_nodes
e_outbound_nodes
f_losses
g	_loss_ids
h_losses_override
i_build_shapes_dict"
_generic_user_object
�
j_inbound_nodes
k_outbound_nodes
l_losses
m	_loss_ids
n_losses_override
o_build_shapes_dict"
_generic_user_object
�
p_inbound_nodes
q_outbound_nodes
r_losses
s	_loss_ids
t_losses_override
u_build_shapes_dict"
_generic_user_object
�
v_inbound_nodes
w_outbound_nodes
x_losses
y	_loss_ids
z_losses_override
{_build_shapes_dict"
_generic_user_object
�
D_kernel
Ebias
|_inbound_nodes
}_outbound_nodes
~_losses
	_loss_ids
�_losses_override
�_build_shapes_dict"
_generic_user_object
�
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids
�_losses_override
�_build_shapes_dict"
_generic_user_object
�
F_kernel
Gbias
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids
�_losses_override
�_build_shapes_dict"
_generic_user_object
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
�
-	capture_0
.	capture_1B�
 __inference_serving_default_8659inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z-	capture_0z.	capture_1
>:< 2&adam/sequential_conv2d_kernel_momentum
>:< 2&adam/sequential_conv2d_kernel_velocity
0:. 2$adam/sequential_conv2d_bias_momentum
0:. 2$adam/sequential_conv2d_bias_velocity
@:> @2(adam/sequential_conv2d_1_kernel_momentum
@:> @2(adam/sequential_conv2d_1_kernel_velocity
2:0@2&adam/sequential_conv2d_1_bias_momentum
2:0@2&adam/sequential_conv2d_1_bias_velocity
7:5
�b�2%adam/sequential_dense_kernel_momentum
7:5
�b�2%adam/sequential_dense_kernel_velocity
0:.�2#adam/sequential_dense_bias_momentum
0:.�2#adam/sequential_dense_bias_velocity
8:6	�2'adam/sequential_dense_1_kernel_momentum
8:6	�2'adam/sequential_dense_1_kernel_velocity
1:/2%adam/sequential_dense_1_bias_momentum
1:/2%adam/sequential_dense_1_bias_velocity
2:0 2sequential/conv2d/kernel
$:" 2sequential/conv2d/bias
4:2 @2sequential/conv2d_1/kernel
&:$@2sequential/conv2d_1/bias
+:)
�b�2sequential/dense/kernel
$:"�2sequential/dense/bias
,:*	�2sequential/dense_1/kernel
%:#2sequential/dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
:2normalization/mean
": 2normalization/variance
:	 2normalization/count
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
trackable_list_wrapper
 "
trackable_dict_wrapper�
__inference___call___8403�
-.@ABCDEFG�
�

�
x 
� "~�{
#
	class_ids�
	class_ids	
'
class_names�
class_names
+
predictions�
predictions�
__inference___call___8565�
-.@ABCDEFG+�(
!�
�
x����������}
� "���
,
	class_ids�
	class_ids���������	
0
class_names!�
class_names���������
4
predictions%�"
predictions����������
 __inference_serving_default_8612i
-.@ABCDEFG8�5
.�+
)�&
inputs���������|�
� "!�
unknown����������
 __inference_serving_default_8659i
-.@ABCDEFG8�5
.�+
)�&
inputs���������|�
� "!�
unknown���������