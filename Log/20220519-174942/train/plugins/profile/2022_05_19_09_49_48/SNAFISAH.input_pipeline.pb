	!?lV?@!?lV?@!!?lV?@	??N??2@??N??2@!??N??2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$!?lV?@aTR'????AV}??b??Y?=yX???*	43333?@2F
Iterator::ModeljM????!?;???T@)bX9????1 ?R?`T@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????ׁ??!iD??8?@)????o??1?Re?A?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ZӼ???!?E?ׂ @)?+e?X??1ʕN???@:Preprocessing2U
Iterator::Model::ParallelMapV2??@??ǈ?!?"=??@)??@??ǈ?1?"=??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipQ?|a2??!?s?? 0@)??_vO??1??8? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???_vO~?!???L???)???_vO~?1???L???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ???v?!?6??s??)Ǻ???v?1?6??s??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<?R??!???R? @)?I+?v?1\?f $??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 18.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s7.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??N??2@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	aTR'????aTR'????!aTR'????      ??!       "      ??!       *      ??!       2	V}??b??V}??b??!V}??b??:      ??!       B      ??!       J	?=yX????=yX???!?=yX???R      ??!       Z	?=yX????=yX???!?=yX???JCPU_ONLYY??N??2@b 