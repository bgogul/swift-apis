import CTensorFlow

// /// The `TF_Operation *` type.
// typealias CTFOperation = OpaquePointer

// class TFGraphBuilder {
//     enum Input {
//         case single(TF_Output)
//         case list([TF_Output])
//     }

//     let graph: CTFGraph = TF_NewGraph()

//     /// A status object to pass to TF graph building operations.
//     private let status: CTFStatus = TF_NewStatus()

//     private var nodeCounter: Int = 0

//     deinit {
//         TF_DeleteGraph(graph)
//         TF_DeleteStatus(status)
//     }

//     func newNodeName(base: String) -> String {
//         let name = "\(base)_\(nodeCounter)"
//         nodeCounter += 1
//         return name
//     }

//     func updateAttribute(
//         _ desc: CTFOperationDescription,
//         _ name: String,
//         _  attrValue: LazyTensorOperation.Attribute) {
//         switch attrValue {
//         case LazyTensorOperation.Attribute.tensorDataTypeValue(let value):
//             TF_SetAttrType(desc, name, value._cDataType)
//         case LazyTensorOperation.Attribute.boolValue(let value):
//             TF_SetAttrBool(desc, name, value ? 1 : 0)
//         case LazyTensorOperation.Attribute.intValue(let value):
//             TF_SetAttrInt(desc, name, Int64(value))
//         case LazyTensorOperation.Attribute.floatValue(let value):
//             TF_SetAttrFloat(desc, name, value)
//         case LazyTensorOperation.Attribute.stringValue(let value): do {
//                 value.utf8CString.withUnsafeBufferPointer { buffer in
//                     // utf8CString is null-terminated; TF_SetAttrString wants
//                     // non-null-terminated.
//                     TF_SetAttrString(
//                         desc, name, buffer.baseAddress, buffer.count - 1)
//                 }
//             }
//         case LazyTensorOperation.Attribute.intArray(let values): do {
//                 let values64 = values.map { Int64($0) }
//                 values64.withUnsafeBufferPointer { buffer in
//                     TF_SetAttrIntList(
//                         desc, name, buffer.baseAddress, Int32(buffer.count))
//                 }
//             }
//         case LazyTensorOperation.Attribute.constTensor(let value): do {
//                 let cTensor = TFE_TensorHandleResolve(
//                     value._cTensorHandle, status)
//                 checkOk(status)
//                 TF_SetAttrTensor(desc, name, cTensor!, status)
//             }
//         case LazyTensorOperation.Attribute.tensorDataTypeArray(let values): do {
//                 values.withUnsafeBufferPointer { buffer in
//                     buffer.withMemoryRebound(to: TF_DataType.self) {
//                         reboundBuffer in
//                         TF_SetAttrTypeList(
//                             desc, name,
//                             reboundBuffer.baseAddress,
//                             Int32(reboundBuffer.count))
//                     }
//                 }
//             }
//         case LazyTensorOperation.Attribute.optionalTensorShapeArray(let values): do {
//                 let flattenedDims = values.flatMap { (tensorShapeOpt) -> [Int64] in
//                     if let tensorShape = tensorShapeOpt {
//                         return tensorShape.dimensions.map(Int64.init)
//                     }
//                     return []
//                 }
//                 let ranks = values.map { shape in (shape?.rank).map(Int32.init) ?? -1 }
//                 flattenedDims.withUnsafeBufferPointer { flattenedDimsBuffer in
//                     var dimsPtr: UnsafePointer<Int64>? = flattenedDimsBuffer.baseAddress
//                     var dims: [UnsafePointer<Int64>?] = []
//                     for rank in ranks {
//                         dims.append(dimsPtr)
//                         if rank >= 0 {
//                             dimsPtr = dimsPtr.map { $0.advanced(by: Int(rank)) }
//                         }
//                     }
//                     dims.withUnsafeMutableBufferPointer { dimsBuffer in
//                         ranks.withUnsafeBufferPointer { ranksBuffer in
//                             TF_SetAttrShapeList(
//                                 desc, name,
//                                 dimsBuffer.baseAddress,
//                                 ranksBuffer.baseAddress,
//                                 Int32(ranksBuffer.count))
//                         }
//                     }
//                 }
//             }
//         default:
//             assert(false, "Unhandled attribute \(name):\(attrValue)")
//         }
//     }

//     func newTFGraphNode(
//         name: String,
//         attrs: [String: LazyTensorOperation.Attribute],
//         inputs: [Input],
//         device: String?
//     ) -> CTFOperation? {
//         // Create a new graph node now.
//         let desc: CTFOperationDescription! = TF_NewOperation(
//             graph, name, newNodeName(base: name))

//         // Set Attributes
//         for (name, value) in attrs {
//             updateAttribute(desc, name, value)
//         }

//         // Add Inputs
//         for input in inputs {
//             switch input {
//             case Input.single(let singleInput):
//                 TF_AddInput(desc, singleInput)
//             case Input.list(let inputList): do {
//                     inputList.withUnsafeBufferPointer { buffer in
//                         TF_AddInputList(desc, buffer.baseAddress, Int32(buffer.count))
//                     }
//                 }
//             }
//         }

//         if let device = device {
//             TF_SetDevice(desc, device)
//         }
//         // Finalize operation.
//         let graphNode = TF_FinishOperation(desc, status)
//         checkOk(status)
//         return graphNode!
//     }

//     func newTFConstNode(_ handle: TFETensorHandle) -> TF_Output {
//         let cTensorHandle = handle._cTensorHandle
//         let cTensor = TFE_TensorHandleResolve(cTensorHandle, status)
//         checkOk(status)
//         let desc = TF_NewOperation(graph, "Const", newNodeName(base: "Const"))
//         checkOk(status)
//         TF_SetAttrType(desc, "dtype", TFE_TensorHandleDataType(cTensorHandle))
//         TF_SetAttrTensor(desc, "value", cTensor, status)
//         checkOk(status)
//         let constNode = TF_FinishOperation(desc, status)
//         return TF_Output(oper: constNode, index: 0)
//     }
// }

// class TFFunctionBuilder {
//     struct FunctionDescription {
//         let function: CTFFunction
//         let outputCount: Int
//         let outputGroups: [Int]
//     }

//     /// A status object to pass to TF graph building operations.
//     private static let status: CTFStatus = TF_NewStatus()

//     static func tfFunction(
//         _ graphDescription: TFGraphDescription,
//         _ tracedFunctionName: String) -> FunctionDescription {
//         let graph = graphDescription.graph
//         let eagerContext = _TFCGetGlobalEagerContext()
//         let inputs = graphDescription.inputs
//         let outputs = graphDescription.outputs
//         let tracedGraphFn = graphDescription.graphNodes.withUnsafeBufferPointer {
//             operations -> CTFFunction in
//             let base = operations.baseAddress
//             let tracedGraphFn = TF_GraphToFunction(graph, tracedFunctionName,
//                 /*append_hash_to_fn_name*/ 1,
//                 /*num_opers*/ Int32(operations.count),
//                 /*opers*/ base,
//                 /*numinputs*/ Int32(inputs.count),
//                 /*inputs*/ inputs,
//                 /*noutputs*/ Int32(outputs.count),
//                 /*outputs*/ outputs,
//                 /*outputnames*/ nil,
//                 /*functionoptions*/ nil, "", status)
//             checkOk(status)
//             if _RuntimeConfig.printsDebugLog {
//                 var len: Int = 0
//                 let funcDebugStr = TF_FunctionDebugString(tracedGraphFn, &len)!
//                 debugLog("The traced function is:\n\(String(cString: funcDebugStr))")
//                 free(funcDebugStr)
//                 debugLog("Corresponding lazy tensor operations:\n")
//                 for output in graphDescription.outputs {
//                     debugLog("  \(output)")
//                 }
//             }
//             return tracedGraphFn!
//         }
//         TFE_ContextAddFunction(eagerContext, tracedGraphFn, status)

//         return FunctionDescription(
//             function: tracedGraphFn,
//             outputCount: outputs.count,
//             outputGroups: graphDescription.outputGroups
//         )
//     }

//     static func execute(
//         _ function: FunctionDescription,
//         _ inputs: [TFETensorHandle]) -> [TFETensorHandle] {
//         let eagerContext = _TFCGetGlobalEagerContext()
//         let fname = TF_FunctionName(function.function)!
//         let eagerOp: CTFEOp! = TFE_NewOp(eagerContext, fname, status)
//         defer { TFE_DeleteOp(eagerOp) }
//         checkOk(status)

//         let deviceName = _ExecutionContext.global.currentDeviceName
//         if let deviceName = deviceName {
//             debugLog("Placing the trace func on device \(deviceName).")
//             TFE_OpSetDevice(eagerOp, deviceName, status)
//             checkOk(status)
//         }

//         for input in inputs {
//             TFE_OpAddInput(eagerOp, input._cTensorHandle, status)
//             checkOk(status)
//         }

//         var returnValues = [CTensorHandle?](
//             repeating: nil, count: function.outputCount)
//         var outputReturnValueCount = Int32(function.outputCount)
//         TFE_Execute(eagerOp, &returnValues, &outputReturnValueCount, status)
//         checkOk(status)

//         return returnValues.map  { TFETensorHandle(_owning: $0!) }
//     }

//     static func removeFunction(_ function: FunctionDescription) {
//         let eagerContext = _TFCGetGlobalEagerContext()
//         let fname = TF_FunctionName(function.function)!
//         TFE_ContextRemoveFunction(eagerContext, fname, status)
//         checkOk(status)
//         TF_DeleteFunction(function.function)
//     }
// }

// class TFGraphDescription {
//     var inputs: [TF_Output] = []
//     var inputValues: [TFETensorHandle] = []
//     var graphNodes: [CTFOperation?] = []
//     var outputs: [TF_Output] = []
//     var outputGroups: [Int] = []

//     var graph: CTFGraph { graphBuilder.graph }
//     var tfFunction: TFFunctionBuilder.FunctionDescription {
//         TFFunctionBuilder.tfFunction(
//             self, graphBuilder.newNodeName(base: "lazyTrace"))
//     }

//     private var graphBuilder = TFGraphBuilder()

//     init(_ desc: LazyTensorTrace) {
//         var graphNodesCache: [ObjectIdentifier: CTFOperation?] = [:]
//         for op in desc.operations {
//             let opInputs = op.inputs.map { input -> TFGraphBuilder.Input in
//                 switch input {
//                 case LazyTensorOperation.Input.single(let h):
//                     return TFGraphBuilder.Input.single(
//                         formTFOutput(h, graphNodesCache))
//                 case LazyTensorOperation.Input.list(let elements):
//                     return TFGraphBuilder.Input.list(elements.map {
//                             formTFOutput($0, graphNodesCache) })
//                 }
//             }
//             let graphNode = graphBuilder.newTFGraphNode(
//                 name: op.name,
//                 attrs: op.attributes,
//                 inputs: opInputs,
//                 device: op.device)
//             let id = ObjectIdentifier(op)
//             graphNodesCache[id] = graphNode
//             if op.name != "Placeholder" {
//                 graphNodes.append(graphNode)
//             }
//         }
//         self.inputs = desc.inputs.map {
//             TF_Output(oper: graphNodesCache[ObjectIdentifier($0)]!, index: 0)
//         }
//         self.inputValues = desc.inputValues
//         for output in desc.outputs {
//             let graphNode = graphNodesCache[ObjectIdentifier(output)]!
//             outputGroups.append(output.outputCount)
//             outputs += Array((0..<output.outputCount).map {
//                     TF_Output(oper: graphNode, index: Int32($0)) })
//         }
//     }

//     private func formTFOutput(
//         _ lazyHandle: LazyTensor,
//         _ graphNodesCache: [ObjectIdentifier: CTFOperation?]) -> TF_Output {
//         if case let LazyTensor.Handle.symbolic(
//             lazyOp, index, _) = lazyHandle.handle {
//             let id = ObjectIdentifier(lazyOp)
//             return TF_Output(oper: graphNodesCache[id]!, index: Int32(index))
//         }
//         assert(false, "Should only have symbolic inputs.")
//     }
// }

extension LazyTensorOperation {
    func materialized(index: Int) -> TFETensorHandle {
        return materialized()[index]
    }

    func materialized() -> [TFETensorHandle] {
        if let outputs = outputs { return outputs }

        LazyTensorOperation.materializeLiveTensors(self)

        // Our outputs should have been updated by now. Otherwise,
        // something terrible happened!
        assert(outputs != nil, "Materialization failed!")
        return outputs!
    }

    func maybeMaterializeInputs() {
        func maybeMaterialized(lazyTensor: LazyTensor) -> LazyTensor {
            let handle = lazyTensor.handle
            if case let LazyTensor.Handle.symbolic(lazyOp, index, _) = handle {
                if let outputs = lazyOp.outputs {
                    return LazyTensor(_materialized: outputs[index])
                }
            }
            return lazyTensor
        }

        func maybeMaterialized(input: Input) -> Input {
            switch input {
            case Input.single(let h):
                return Input.single(maybeMaterialized(lazyTensor: h))
            case LazyTensorOperation.Input.list(let elements):
                return Input.list(elements.map {
                        maybeMaterialized(lazyTensor: $0) })
            }
        }
        inputs = inputs.map { maybeMaterialized(input: $0) }
    }


    static var fcount = 0
    private static func materializeLiveTensors(_ lazyOp: LazyTensorOperation) {
        LazyTensor._materializationCallback("lazy")
        let lazyTrace = LazyTensorTrace(lazyOp)
        if _RuntimeConfig.printsDebugLog {
            print("\(fcount): lazyDescription.nameWithid")
            print("\(lazyTrace)")
            fcount += 1
        }
        LazyTensor._materializationCallback("graphdesc")
        // let graph = TFGraph(lazyTrace)
        let function = TFFunction(lazyTrace)
        debugLog("The traced function is \(function))")
        LazyTensor._materializationCallback("tffunction")
        let allOutputs = function.execute(lazyTrace.inputValues)
        LazyTensor._materializationCallback("execute")

        // Slice up the outputs to various lazy tensors
        var start: Int = 0
        for lazyOp in lazyTrace.originalOutputs {
            let end = start + lazyOp.outputCount
            lazyOp.outputs = Array(allOutputs[start..<end])
            start = end
        }

        // On all the live operations rewrite the inputs so that we drop references
        // to the LazyTensorOperations..
        LazyTensor.forEachOperation { $0.maybeMaterializeInputs() }
        // Cleanup
        // TFFunctionBuilder.removeFunction(function)
    }
}
