	46<?R??46<?R??!46<?R??	T?3@T?3@!T?3@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$46<?R???? ???A>yX?5???Y46<?R??*	433333a@2F
Iterator::Modelh??s???!⎸#?8C@)??ܥ?1????/?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateΈ?????!⎸#?;@)	?^)ˠ?16eMYS?7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?:pΈ??!$??;?N:@)2U0*???1?qG??6@:Preprocessing2U
Iterator::Model::ParallelMapV2??ZӼ???!*kʚ??@)??ZӼ???1*kʚ??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???1段?!qG??N@)?5?;Nс?1???)kJ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?s?!????/?@)a2U0*?s?1????/?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice/n??r?!fMYS֔	@)/n??r?1fMYS֔	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap8??d?`??!lʚ???<@)??_?Le?1??#??;??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 10.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9T?3@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?? ????? ???!?? ???      ??!       "      ??!       *      ??!       2	>yX?5???>yX?5???!>yX?5???:      ??!       B      ??!       J	46<?R??46<?R??!46<?R??R      ??!       Z	46<?R??46<?R??!46<?R??JCPU_ONLYYT?3@b 