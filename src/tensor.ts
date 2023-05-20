import type { Device, Deviceish } from "./device";
import { getDevice } from "./devices";
import { Shape, Strides, defaultStrides, shapeSize } from "./shape";
import { ones } from "./factories";
import { ATypedArray, Dtype, getDtype } from "./dtype";
import {
    TensorArrayData,
    UntypedStorage,
    newTypedArrayFromArray,
} from "./storage";
import type { GradientFunction, GradientContext } from "./autograd";
import type { KernelConfigInput, KernelParamsInput } from "./kernel";
import * as ops from "./ops_opgen";
import * as aops from "./ops_artisanal";
import { TensorBase } from "./tensor_base";
import { product,range } from "./utils";

export type TensorData = TensorArrayData | ATypedArray | UntypedStorage;

export type TensorSpec = {
    data: TensorData;
    dtype?: Dtype;
    requiresGrad?: boolean;
    device?: Deviceish;
    shape?: Shape;
    strides?: Strides;
};

export class Tensor extends TensorBase {
    private _device: Device;

    private _storage: UntypedStorage;
    private _dtype: Dtype;
    private _shape: Shape;
    private _strides: Strides;

    private _requiresGrad: boolean = false;
    private _gradFunc: GradientFunction | null;
    private _gradCtx: GradientContext | null;
    public grad: Tensor | null = null;

    get storage(): UntypedStorage {
        return this._storage;
    }
    get dtype(): Dtype {
        return this._dtype;
    }
    get shape(): Shape {
        return this._shape;
    }
    get strides(): Strides {
        return this._strides;
    }
    get device(): Device {
        return this._device;
    }
    get isContiguous(): boolean {
        let strides = this.strides;
        let shape = this.shape;
        let offset = 1;
        for (let i = shape.length - 1; i >= 0; i--) {
            if (strides[i] !== offset) {
                return false;
            }
            offset *= shape[i];
        }
        return true;
    }
    private get isScalar(): boolean {
        return (
            this.shape.length === 0 ||
            (this.shape.length === 1 && this.shape[0] === 1)
        );
    }

    get requiresGrad(): boolean {
        return this._requiresGrad;
    }
    set requiresGrad(value: boolean) {
        if (this._gradFunc) {
            throw new Error(
                "You can only change requiresGrad flags of leaf variables. If you want to use a computed variable in a subgraph that doesn't require differentiation use valueNoGrad = value.detach()."
            );
        }
        this._requiresGrad = value;
    }
    get gradFunc(): GradientFunction | null {
        return this._gradFunc;
    }

    constructor(
        arrayOrSpec: TensorData | TensorSpec,
        dtype?: Dtype,
        device?: Deviceish,
        requiresGrad?: boolean
    ) {
        super();
        let d = getDevice(device);
        let dt = getDtype(dtype);
        if (arrayOrSpec === null) {
            throw new Error("Cannot create tensor from null");
        } else if (arrayOrSpec instanceof Array) {
            const array = newTypedArrayFromArray(arrayOrSpec, dt, d);
            this._storage = array.storage;
            this._dtype = dt;
            this._shape = array.shape;
            this._strides = array.strides;
        } else if (arrayOrSpec.hasOwnProperty("data")) {
            const jdata = arrayOrSpec as TensorSpec;
            dt = jdata.dtype ? getDtype(jdata.dtype) : dt;
            requiresGrad = requiresGrad || jdata.requiresGrad;
            if (jdata.data instanceof Array) {
                const array = newTypedArrayFromArray(jdata.data, dt, d);
                this._storage = array.storage;
                this._dtype = dt;
                this._shape = array.shape;
                this._strides = array.strides;
            } else if (jdata.data instanceof UntypedStorage) {
                this._storage = jdata.data;
                this._dtype = dt;
                if (jdata.shape === undefined && jdata.strides === undefined) {
                    throw new Error("Cannot create tensor from storage without also specifying the shape and strides.");
                }
                this._shape = jdata.shape || [];
                this._strides = jdata.strides || defaultStrides(this._shape);
            } else {
                throw new Error("Cannot create tensor from json data " + jdata);
            }
        } else {
            throw new Error(
                "Invalid data type for Tensor constructor. Expected an array of values or a json object with a 'data' property."
            );
        }
        this._device = d;
        this._requiresGrad = requiresGrad || false;
        this._gradFunc = null;
        this._gradCtx = null;
        this.grad = null;
    }
    withShape(shape: Shape, strides: Strides): Tensor {
        if (shapeSize(shape) != shapeSize(this.shape)) {
            throw new Error(
                `Cannot reshape tensor of size ${this.shape} to ${shape}`
            );
        }
        return new Tensor({
            data: this.storage,
            dtype: this.dtype,
            device: this.device,
            requiresGrad: this.requiresGrad,
            shape,
            strides,
        });
    }

    get [Symbol.toStringTag]() {
        return "Tensor";
    }
    toString(): string {
        let rg = this.requiresGrad ? ", requiresGrad=true" : "";
        if (this._gradFunc) {
            rg = ", gradFunc";
        }
        return `tensor([${this.shape}], ${this.dtype}${rg})`;
    }
    async toArrayAsync(): Promise<TensorArrayData> {
        await this.storage.mapReadAsync();
        const data = this.storage.getTypedArray(this.dtype);
        const shape = this.shape;
        const strides = this.strides;
        if (shape.length == 0 || (shape.length == 1 && shape[0] == 1)) {
            return [data[0]];
        }
        const index: number[] = [];
        return readArray(index);
        function readArray(index: number[]): TensorArrayData {
            const dim = index.length;
            // console.log("Read array: ", index, "dim=", dim);
            if (dim == shape.length - 1) {
                const offset = index.reduce(
                    (acc, cur, i) => acc + cur * strides[i],
                    0
                );
                // console.log("offset=", offset);
                const length = shape[dim];
                // console.log("length=", length);
                const subarray = data.subarray(offset, offset + length);
                // console.log("subarray=", subarray);
                const ar = Array.from(subarray);
                // console.log("ar=", ar);
                return ar;
            } else {
                const result: TensorArrayData = [];
                for (let i = 0; i < shape[dim]; i++) {
                    index.push(i);
                    result.push(readArray(index));
                    index.pop();
                }
                return result;
            }
        }
    }

    runKernelInplace(
        name: string,
        config: KernelConfigInput,
        params: KernelParamsInput,
        ...additionalInputs: Tensor[]
    ): Tensor {
        if (this.requiresGrad) {
            throw new Error(`A tensor that requires a gradient cannot be used in an in-place operation`);
        }
        const d = this.device;
        const kernel = this.device.getKernel(name, config);
        const inputBuffers = additionalInputs.map((t) =>
            d.getBufferForKernel(t.storage, t.dtype)
        );
        const outputBuffers = [d.getBufferForKernel(this.storage, this.dtype)];
        kernel.run(inputBuffers, params, outputBuffers);
        return this;
    }
    runKernel(
        name: string,
        config: KernelConfigInput,
        params: KernelParamsInput,
        outputShapes: Shape[],
        ...additionalInputs: Tensor[]
    ): Tensor[] {
        const d = this.device;
        const kernel = d.getKernel(name, config);
        const inputBuffers = [
            d.getBufferForKernel(this.storage, this.dtype),
            ...additionalInputs.map((t) =>
                d.getBufferForKernel(t.storage, t.dtype)
            ),
        ];
        const outputBuffers = kernel.run(inputBuffers, params) as (
            | GPUBuffer
            | ATypedArray
        )[];
        if (outputBuffers.length !== outputShapes.length) {
            throw new Error(
                `Expected ${outputShapes.length} output buffers (given the provided outputShapes to runKernel), but got ${outputBuffers.length} output buffers when running the kernel "${name}".`
            );
        }
        return outputBuffers.map(
            (outputBuffer, i) =>
                new Tensor({
                    data: this.device.getStorageFromKernel(outputBuffer),
                    dtype: this.dtype,
                    shape: outputShapes[i],
                    strides: defaultStrides(outputShapes[i]),
                    device: this.device,
                })
        );
    }

    async broadcastTo(targetShape: Shape): Promise<Tensor> {
        const inputShape = this.shape;
        const inputRank = inputShape.length;
        const targetRank = targetShape.length;

        if (inputRank > targetRank) {
            throw new Error("Cannot broadcast to a smaller number of dimensions.");
        }
        let expandedShape = inputShape.slice();
        let newArray:any = await this.toArrayAsync();
        console.log(newArray)
        
        if(targetRank >= 3){
            throw new Error("Broadcasting to more than 2 dimensions is not supported yet.")
        }

        // Rule 1: Expand dimensions without adding new ones
        if(inputRank === targetRank){
            if(targetRank === 2 && inputShape[0] === 1){
                expandedShape[0] = targetShape[0]
                for(let i=1;i<targetShape[0];i++){
                    newArray[i] = newArray[0].slice()
                }
            }

        console.log(expandedShape,newArray)
        return new Tensor(newArray, this.dtype, this.device,this.requiresGrad);
    }

    // expand dimensions
        let firstDim = shapeSize(targetShape) / shapeSize(inputShape);
        expandedShape = [firstDim, ...expandedShape];
        for(let i=1;i<firstDim;i++){
            newArray[i] = newArray[0];
        }
    
        return new Tensor(newArray, this.dtype, this.device,this.requiresGrad);
    }
    

    detach(): Tensor {
        if (this._requiresGrad || this._gradFunc) {
            return new Tensor({
                data: this.storage,
                dtype: this.dtype,
                requiresGrad: false,
                shape: this.shape,
                strides: this.strides,
                device: this.device,
            });
        }
        return this;
    }

    setGradientFunction(
        ctx: GradientContext,
        gradFunc: GradientFunction
    ): void {
        this._gradFunc = gradFunc;
        this._gradCtx = ctx;
        this._requiresGrad = true;
    }

    backward(gradient?: Tensor): void {
        if (!this.requiresGrad) {
            throw new Error(`Cannot call backward on a tensor that does not require gradients`);
        }
        let grad: Tensor;
        if (gradient) {
            grad = gradient;
        } else {
            if (!this.isScalar) {
                throw new Error(
                    `Gradient can only be implicitly created for scalar outputs`
                );
            }
            grad = ones(1);
        }
        if (this.grad) {
            this.grad.add_(grad);
        } else {
            this.grad = grad;
        }
        if (!this._gradFunc || !this._gradCtx) {
            return;
        }
        const grads = this._gradFunc(this._gradCtx, grad);
        const inputs = this._gradCtx.inputsWithGradient;
        for (let i = 0; i < inputs.length; i++) {
            const input = inputs[i];
            if (input === null) {
                continue;
            }
            const grad = grads[i];
            if (grad) {
                input.backward(grad);
            } else {
                throw new Error(
                    `Gradient function did not return a gradient for input #${i} (out of ${inputs.length}). ${grads.length} gradients were returned.`
                );
            }
        }
    }

    expand(shape: Shape): Tensor {
        const newShape = shape.slice();
        const newStrides = Array(newShape.length).fill(0);
        // Update newStrides based on the current strides
        // so that the expansion happens
        // in the correct direction
        let j = newShape.length - 1;
        let thisShape = this.shape;
        let thisStrides = this.strides;
        for (let i = thisShape.length - 1; i >= 0; i--) {
            if (thisShape[i] === 1) {
                newStrides[j] = 0;
            } else {
                newStrides[j] = thisStrides[i];
                j--;
            }
            if (newShape[j] === -1) {
                newShape[j] = thisShape[i];
            }
        }
        // console.log("EXPAND", this.shape, this.strides, shape, newShape, newStrides);
        return this.withShape(newShape, newStrides);
    }
    mm(other: Tensor): Tensor {
        return aops.mm(this, other);
    }
    t(): Tensor {
        return aops.t(this);
    }
    zero_(): Tensor {
        throw new Error("Tensor zero_ is not implemented");
    }

    // Codegen marker
    /**
    * ![Plot of abs and its gradient](/plots/abs.svg)
    *
    * Calculates:
    * ```js
    * output = abs(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = input == 0 ? 0 : (input > 0 ? outputGrad : -outputGrad)
    * ```
    *
    * @returns the output tensor
    */
    abs(): Tensor {
        return ops.abs(this);
    }
    /**
    * Alias for `abs`.
    *
    * ![Plot of abs and its gradient](/plots/abs.svg)
    *
    * Calculates:
    * ```js
    * output = abs(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = input == 0 ? 0 : (input > 0 ? outputGrad : -outputGrad)
    * ```
    *
    * @returns the output tensor
    */
    absolute(): Tensor {
        return ops.abs(this);
    }
    /**
    * ![Plot of abs and its gradient](/plots/abs.svg)
    *
    * Calculates:
    * ```js
    * output = abs(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = input == 0 ? 0 : (input > 0 ? outputGrad : -outputGrad)
    * ```
    *
    * @returns the output tensor
    */
    abs_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("abs_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of acos and its gradient](/plots/acos.svg)
    *
    * Calculates:
    * ```js
    * output = acos(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad / sqrt(1 - input * input)
    * ```
    *
    * @returns the output tensor
    */
    acos(): Tensor {
        return ops.acos(this);
    }
    /**
    * Alias for `acos`.
    *
    * ![Plot of acos and its gradient](/plots/acos.svg)
    *
    * Calculates:
    * ```js
    * output = acos(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad / sqrt(1 - input * input)
    * ```
    *
    * @returns the output tensor
    */
    arccos(): Tensor {
        return ops.acos(this);
    }
    /**
    * ![Plot of acos and its gradient](/plots/acos.svg)
    *
    * Calculates:
    * ```js
    * output = acos(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad / sqrt(1 - input * input)
    * ```
    *
    * @returns the output tensor
    */
    acos_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("acos_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of acosh and its gradient](/plots/acosh.svg)
    *
    * Calculates:
    * ```js
    * output = acosh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(input * input - 1)
    * ```
    *
    * @returns the output tensor
    */
    acosh(): Tensor {
        return ops.acosh(this);
    }
    /**
    * Alias for `acosh`.
    *
    * ![Plot of acosh and its gradient](/plots/acosh.svg)
    *
    * Calculates:
    * ```js
    * output = acosh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(input * input - 1)
    * ```
    *
    * @returns the output tensor
    */
    arccosh(): Tensor {
        return ops.acosh(this);
    }
    /**
    * ![Plot of acosh and its gradient](/plots/acosh.svg)
    *
    * Calculates:
    * ```js
    * output = acosh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(input * input - 1)
    * ```
    *
    * @returns the output tensor
    */
    acosh_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("acosh_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = input + other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad; otherGrad = outputGrad
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @param alpha the alpha value to multiply `other` with
    * @returns the output tensor
    */
    add(other: Tensor, alpha?: number): Tensor {
        return ops.add(this, other, alpha);
    }
    /**
    * Calculates:
    * ```js
    * output = input + other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad; otherGrad = outputGrad
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @param alpha the alpha value to multiply `other` with
    * @returns the output tensor
    */
    add_(other: Tensor, alpha?: number): Tensor {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernelInplace("add_", { dtype: this.dtype }, params, other);
    }
    /**
    * ![Plot of asin and its gradient](/plots/asin.svg)
    *
    * Calculates:
    * ```js
    * output = asin(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(1 - input * input)
    * ```
    *
    * @returns the output tensor
    */
    asin(): Tensor {
        return ops.asin(this);
    }
    /**
    * Alias for `asin`.
    *
    * ![Plot of asin and its gradient](/plots/asin.svg)
    *
    * Calculates:
    * ```js
    * output = asin(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(1 - input * input)
    * ```
    *
    * @returns the output tensor
    */
    arcsin(): Tensor {
        return ops.asin(this);
    }
    /**
    * ![Plot of asin and its gradient](/plots/asin.svg)
    *
    * Calculates:
    * ```js
    * output = asin(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(1 - input * input)
    * ```
    *
    * @returns the output tensor
    */
    asin_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("asin_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of asinh and its gradient](/plots/asinh.svg)
    *
    * Calculates:
    * ```js
    * output = asinh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(input * input + 1)
    * ```
    *
    * @returns the output tensor
    */
    asinh(): Tensor {
        return ops.asinh(this);
    }
    /**
    * Alias for `asinh`.
    *
    * ![Plot of asinh and its gradient](/plots/asinh.svg)
    *
    * Calculates:
    * ```js
    * output = asinh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(input * input + 1)
    * ```
    *
    * @returns the output tensor
    */
    arcsinh(): Tensor {
        return ops.asinh(this);
    }
    /**
    * ![Plot of asinh and its gradient](/plots/asinh.svg)
    *
    * Calculates:
    * ```js
    * output = asinh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / sqrt(input * input + 1)
    * ```
    *
    * @returns the output tensor
    */
    asinh_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("asinh_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of atan and its gradient](/plots/atan.svg)
    *
    * Calculates:
    * ```js
    * output = atan(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input * input + 1)
    * ```
    *
    * @returns the output tensor
    */
    atan(): Tensor {
        return ops.atan(this);
    }
    /**
    * Alias for `atan`.
    *
    * ![Plot of atan and its gradient](/plots/atan.svg)
    *
    * Calculates:
    * ```js
    * output = atan(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input * input + 1)
    * ```
    *
    * @returns the output tensor
    */
    arctan(): Tensor {
        return ops.atan(this);
    }
    /**
    * ![Plot of atan and its gradient](/plots/atan.svg)
    *
    * Calculates:
    * ```js
    * output = atan(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input * input + 1)
    * ```
    *
    * @returns the output tensor
    */
    atan_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("atan_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = atan2(input, other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * other / (input * input + other * other); otherGrad = -outputGrad * input / (input * input + other * other)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    atan2(other: Tensor): Tensor {
        return ops.atan2(this, other);
    }
    /**
    * Alias for `atan2`.
    *
    * Calculates:
    * ```js
    * output = atan2(input, other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * other / (input * input + other * other); otherGrad = -outputGrad * input / (input * input + other * other)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    arctan2(other: Tensor): Tensor {
        return ops.atan2(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = atan2(input, other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * other / (input * input + other * other); otherGrad = -outputGrad * input / (input * input + other * other)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    atan2_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("atan2_", { dtype: this.dtype }, params, other);
    }
    /**
    * ![Plot of ceil and its gradient](/plots/ceil.svg)
    *
    * Calculates:
    * ```js
    * output = ceil(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    ceil(): Tensor {
        return ops.ceil(this);
    }
    /**
    * ![Plot of ceil and its gradient](/plots/ceil.svg)
    *
    * Calculates:
    * ```js
    * output = ceil(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    ceil_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("ceil_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = other >= 0 ? abs(input) : -abs(input)
    * ```
    *
    * Gradient:
    * ```js
    * var dir = other >= 0 ? (input >= 0 ? 1.0 : -1.0) : (input >= 0 ? -1.0 : 1.0); inputGrad = input == 0.0 ? 0.0 : outputGrad * dir; otherGrad = 0
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    copysign(other: Tensor): Tensor {
        return ops.copysign(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = other >= 0 ? abs(input) : -abs(input)
    * ```
    *
    * Gradient:
    * ```js
    * var dir = other >= 0 ? (input >= 0 ? 1.0 : -1.0) : (input >= 0 ? -1.0 : 1.0); inputGrad = input == 0.0 ? 0.0 : outputGrad * dir; otherGrad = 0
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    copysign_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("copysign_", { dtype: this.dtype }, params, other);
    }
    /**
    * ![Plot of cos and its gradient](/plots/cos.svg)
    *
    * Calculates:
    * ```js
    * output = cos(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad * sin(input)
    * ```
    *
    * @returns the output tensor
    */
    cos(): Tensor {
        return ops.cos(this);
    }
    /**
    * ![Plot of cos and its gradient](/plots/cos.svg)
    *
    * Calculates:
    * ```js
    * output = cos(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad * sin(input)
    * ```
    *
    * @returns the output tensor
    */
    cos_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("cos_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of cosh and its gradient](/plots/cosh.svg)
    *
    * Calculates:
    * ```js
    * output = cosh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * sinh(input)
    * ```
    *
    * @returns the output tensor
    */
    cosh(): Tensor {
        return ops.cosh(this);
    }
    /**
    * ![Plot of cosh and its gradient](/plots/cosh.svg)
    *
    * Calculates:
    * ```js
    * output = cosh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * sinh(input)
    * ```
    *
    * @returns the output tensor
    */
    cosh_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("cosh_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of deg2rad and its gradient](/plots/deg2rad.svg)
    *
    * Calculates:
    * ```js
    * output = input * 0.017453292519943295
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * 0.017453292519943295
    * ```
    *
    * @returns the output tensor
    */
    deg2rad(): Tensor {
        return ops.deg2rad(this);
    }
    /**
    * ![Plot of deg2rad and its gradient](/plots/deg2rad.svg)
    *
    * Calculates:
    * ```js
    * output = input * 0.017453292519943295
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * 0.017453292519943295
    * ```
    *
    * @returns the output tensor
    */
    deg2rad_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("deg2rad_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = input / other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / other; otherGrad = -outputGrad * input / (other * other)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @param alpha the alpha value to multiply `other` with
    * @returns the output tensor
    */
    div(other: Tensor, alpha?: number): Tensor {
        return ops.div(this, other, alpha);
    }
    /**
    * Alias for `div`.
    *
    * Calculates:
    * ```js
    * output = input / other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / other; otherGrad = -outputGrad * input / (other * other)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @param alpha the alpha value to multiply `other` with
    * @returns the output tensor
    */
    divide(other: Tensor, alpha?: number): Tensor {
        return ops.div(this, other, alpha);
    }
    /**
    * Calculates:
    * ```js
    * output = input / other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / other; otherGrad = -outputGrad * input / (other * other)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @param alpha the alpha value to multiply `other` with
    * @returns the output tensor
    */
    div_(other: Tensor, alpha?: number): Tensor {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernelInplace("div_", { dtype: this.dtype }, params, other);
    }
    /**
    * ![Plot of exp and its gradient](/plots/exp.svg)
    *
    * Calculates:
    * ```js
    * output = exp(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * exp(input)
    * ```
    *
    * @returns the output tensor
    */
    exp(): Tensor {
        return ops.exp(this);
    }
    /**
    * ![Plot of exp and its gradient](/plots/exp.svg)
    *
    * Calculates:
    * ```js
    * output = exp(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * exp(input)
    * ```
    *
    * @returns the output tensor
    */
    exp_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("exp_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of exp2 and its gradient](/plots/exp2.svg)
    *
    * Calculates:
    * ```js
    * output = exp2(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * exp2(input) * 0.6931471805599453
    * ```
    *
    * @returns the output tensor
    */
    exp2(): Tensor {
        return ops.exp2(this);
    }
    /**
    * ![Plot of exp2 and its gradient](/plots/exp2.svg)
    *
    * Calculates:
    * ```js
    * output = exp2(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * exp2(input) * 0.6931471805599453
    * ```
    *
    * @returns the output tensor
    */
    exp2_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("exp2_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of expm1 and its gradient](/plots/expm1.svg)
    *
    * Calculates:
    * ```js
    * output = exp(input) - 1.0
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * exp(input)
    * ```
    *
    * @returns the output tensor
    */
    expm1(): Tensor {
        return ops.expm1(this);
    }
    /**
    * ![Plot of expm1 and its gradient](/plots/expm1.svg)
    *
    * Calculates:
    * ```js
    * output = exp(input) - 1.0
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * exp(input)
    * ```
    *
    * @returns the output tensor
    */
    expm1_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("expm1_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of floor and its gradient](/plots/floor.svg)
    *
    * Calculates:
    * ```js
    * output = floor(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    floor(): Tensor {
        return ops.floor(this);
    }
    /**
    * ![Plot of floor and its gradient](/plots/floor.svg)
    *
    * Calculates:
    * ```js
    * output = floor(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    floor_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("floor_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of frac and its gradient](/plots/frac.svg)
    *
    * Calculates:
    * ```js
    * output = input >= 0.0 ? fract(input) : -fract(-input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad
    * ```
    *
    * @returns the output tensor
    */
    frac(): Tensor {
        return ops.frac(this);
    }
    /**
    * ![Plot of frac and its gradient](/plots/frac.svg)
    *
    * Calculates:
    * ```js
    * output = input >= 0.0 ? fract(input) : -fract(-input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad
    * ```
    *
    * @returns the output tensor
    */
    frac_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("frac_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = sqrt(input * input + other * other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * input / sqrt(input * input + other * other); otherGrad = outputGrad * other / sqrt(input * input + other * other)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    hypot(other: Tensor): Tensor {
        return ops.hypot(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = sqrt(input * input + other * other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * input / sqrt(input * input + other * other); otherGrad = outputGrad * other / sqrt(input * input + other * other)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    hypot_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("hypot_", { dtype: this.dtype }, params, other);
    }
    /**
    * Calculates:
    * ```js
    * output = input * pow(2.0, other)
    * ```
    *
    * Gradient:
    * ```js
    * var out = pow(2.0, other); inputGrad = outputGrad * out; otherGrad = outputGrad * input * out * 0.6931471805599453
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    ldexp(other: Tensor): Tensor {
        return ops.ldexp(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = input * pow(2.0, other)
    * ```
    *
    * Gradient:
    * ```js
    * var out = pow(2.0, other); inputGrad = outputGrad * out; otherGrad = outputGrad * input * out * 0.6931471805599453
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    ldexp_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("ldexp_", { dtype: this.dtype }, params, other);
    }
    /**
    * ![Plot of log and its gradient](/plots/log.svg)
    *
    * Calculates:
    * ```js
    * output = log(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / input
    * ```
    *
    * @returns the output tensor
    */
    log(): Tensor {
        return ops.log(this);
    }
    /**
    * ![Plot of log and its gradient](/plots/log.svg)
    *
    * Calculates:
    * ```js
    * output = log(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / input
    * ```
    *
    * @returns the output tensor
    */
    log_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of log10 and its gradient](/plots/log10.svg)
    *
    * Calculates:
    * ```js
    * output = log(input) * 0.4342944819032518
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input * 2.302585092994046)
    * ```
    *
    * @returns the output tensor
    */
    log10(): Tensor {
        return ops.log10(this);
    }
    /**
    * ![Plot of log10 and its gradient](/plots/log10.svg)
    *
    * Calculates:
    * ```js
    * output = log(input) * 0.4342944819032518
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input * 2.302585092994046)
    * ```
    *
    * @returns the output tensor
    */
    log10_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log10_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of log1p and its gradient](/plots/log1p.svg)
    *
    * Calculates:
    * ```js
    * output = log(input + 1.0)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input + 1.0)
    * ```
    *
    * @returns the output tensor
    */
    log1p(): Tensor {
        return ops.log1p(this);
    }
    /**
    * ![Plot of log1p and its gradient](/plots/log1p.svg)
    *
    * Calculates:
    * ```js
    * output = log(input + 1.0)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input + 1.0)
    * ```
    *
    * @returns the output tensor
    */
    log1p_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log1p_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of log2 and its gradient](/plots/log2.svg)
    *
    * Calculates:
    * ```js
    * output = log2(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input * 0.6931471805599453)
    * ```
    *
    * @returns the output tensor
    */
    log2(): Tensor {
        return ops.log2(this);
    }
    /**
    * ![Plot of log2 and its gradient](/plots/log2.svg)
    *
    * Calculates:
    * ```js
    * output = log2(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (input * 0.6931471805599453)
    * ```
    *
    * @returns the output tensor
    */
    log2_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("log2_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = log(exp(input) + exp(other))
    * ```
    *
    * Gradient:
    * ```js
    * var ein = exp(input); var eoth = exp(other); var addeinv = outputGrad/(ein + eoth); inputGrad = addeinv * ein; otherGrad = addeinv * eoth
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    logaddexp(other: Tensor): Tensor {
        return ops.logaddexp(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = log(exp(input) + exp(other))
    * ```
    *
    * Gradient:
    * ```js
    * var ein = exp(input); var eoth = exp(other); var addeinv = outputGrad/(ein + eoth); inputGrad = addeinv * ein; otherGrad = addeinv * eoth
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    logaddexp_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("logaddexp_", { dtype: this.dtype }, params, other);
    }
    /**
    * Calculates:
    * ```js
    * output = log2(exp2(input) + exp2(other))
    * ```
    *
    * Gradient:
    * ```js
    * var ein = exp2(input); var eoth = exp2(other); var sum_ein_eoth = ein + eoth; inputGrad = outputGrad * (ein / sum_ein_eoth); otherGrad = outputGrad * (eoth / sum_ein_eoth );
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    logaddexp2(other: Tensor): Tensor {
        return ops.logaddexp2(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = log2(exp2(input) + exp2(other))
    * ```
    *
    * Gradient:
    * ```js
    * var ein = exp2(input); var eoth = exp2(other); var sum_ein_eoth = ein + eoth; inputGrad = outputGrad * (ein / sum_ein_eoth); otherGrad = outputGrad * (eoth / sum_ein_eoth );
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    logaddexp2_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("logaddexp2_", { dtype: this.dtype }, params, other);
    }
    /**
    * Calculates:
    * ```js
    * output = input * other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * other; otherGrad = outputGrad * input
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @param alpha the alpha value to multiply `other` with
    * @returns the output tensor
    */
    mul(other: Tensor, alpha?: number): Tensor {
        return ops.mul(this, other, alpha);
    }
    /**
    * Alias for `mul`.
    *
    * Calculates:
    * ```js
    * output = input * other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * other; otherGrad = outputGrad * input
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @param alpha the alpha value to multiply `other` with
    * @returns the output tensor
    */
    multiply(other: Tensor, alpha?: number): Tensor {
        return ops.mul(this, other, alpha);
    }
    /**
    * Calculates:
    * ```js
    * output = input * other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * other; otherGrad = outputGrad * input
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @param alpha the alpha value to multiply `other` with
    * @returns the output tensor
    */
    mul_(other: Tensor, alpha?: number): Tensor {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernelInplace("mul_", { dtype: this.dtype }, params, other);
    }
    /**
    * ![Plot of neg and its gradient](/plots/neg.svg)
    *
    * Calculates:
    * ```js
    * output = -input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad
    * ```
    *
    * @returns the output tensor
    */
    neg(): Tensor {
        return ops.neg(this);
    }
    /**
    * Alias for `neg`.
    *
    * ![Plot of neg and its gradient](/plots/neg.svg)
    *
    * Calculates:
    * ```js
    * output = -input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad
    * ```
    *
    * @returns the output tensor
    */
    negative(): Tensor {
        return ops.neg(this);
    }
    /**
    * ![Plot of neg and its gradient](/plots/neg.svg)
    *
    * Calculates:
    * ```js
    * output = -input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad
    * ```
    *
    * @returns the output tensor
    */
    neg_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("neg_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of positive and its gradient](/plots/positive.svg)
    *
    * Calculates:
    * ```js
    * output = input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad
    * ```
    *
    * @returns the output tensor
    */
    positive(): Tensor {
        return ops.positive(this);
    }
    /**
    * ![Plot of positive and its gradient](/plots/positive.svg)
    *
    * Calculates:
    * ```js
    * output = input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad
    * ```
    *
    * @returns the output tensor
    */
    positive_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("positive_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = pow(input, other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * other * pow(input, other - 1.0); otherGrad = outputGrad * pow(input, other) * log(input)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    pow(other: Tensor): Tensor {
        return ops.pow(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = pow(input, other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * other * pow(input, other - 1.0); otherGrad = outputGrad * pow(input, other) * log(input)
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    pow_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("pow_", { dtype: this.dtype }, params, other);
    }
    /**
    * ![Plot of rad2deg and its gradient](/plots/rad2deg.svg)
    *
    * Calculates:
    * ```js
    * output = input * 57.29577951308232
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * 57.29577951308232
    * ```
    *
    * @returns the output tensor
    */
    rad2deg(): Tensor {
        return ops.rad2deg(this);
    }
    /**
    * ![Plot of rad2deg and its gradient](/plots/rad2deg.svg)
    *
    * Calculates:
    * ```js
    * output = input * 57.29577951308232
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * 57.29577951308232
    * ```
    *
    * @returns the output tensor
    */
    rad2deg_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("rad2deg_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of reciprocal and its gradient](/plots/reciprocal.svg)
    *
    * Calculates:
    * ```js
    * output = 1.0 / input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad / (input * input)
    * ```
    *
    * @returns the output tensor
    */
    reciprocal(): Tensor {
        return ops.reciprocal(this);
    }
    /**
    * ![Plot of reciprocal and its gradient](/plots/reciprocal.svg)
    *
    * Calculates:
    * ```js
    * output = 1.0 / input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad / (input * input)
    * ```
    *
    * @returns the output tensor
    */
    reciprocal_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("reciprocal_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of relu and its gradient](/plots/relu.svg)
    *
    * Calculates:
    * ```js
    * output = max(input, 0.0)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = input > 0.0 ? outputGrad : 0.0
    * ```
    *
    * @returns the output tensor
    */
    relu(): Tensor {
        return ops.relu(this);
    }
    /**
    * ![Plot of relu and its gradient](/plots/relu.svg)
    *
    * Calculates:
    * ```js
    * output = max(input, 0.0)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = input > 0.0 ? outputGrad : 0.0
    * ```
    *
    * @returns the output tensor
    */
    relu_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("relu_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of round and its gradient](/plots/round.svg)
    *
    * Calculates:
    * ```js
    * output = round(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    round(): Tensor {
        return ops.round(this);
    }
    /**
    * ![Plot of round and its gradient](/plots/round.svg)
    *
    * Calculates:
    * ```js
    * output = round(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    round_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("round_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of rsqrt and its gradient](/plots/rsqrt.svg)
    *
    * Calculates:
    * ```js
    * output = 1.0 / sqrt(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad / (2.0 * sqrt(input) * input)
    * ```
    *
    * @returns the output tensor
    */
    rsqrt(): Tensor {
        return ops.rsqrt(this);
    }
    /**
    * ![Plot of rsqrt and its gradient](/plots/rsqrt.svg)
    *
    * Calculates:
    * ```js
    * output = 1.0 / sqrt(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = -outputGrad / (2.0 * sqrt(input) * input)
    * ```
    *
    * @returns the output tensor
    */
    rsqrt_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("rsqrt_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of sigmoid and its gradient](/plots/sigmoid.svg)
    *
    * Calculates:
    * ```js
    * output = 1.0 / (1.0 + exp(-input))
    * ```
    *
    * Gradient:
    * ```js
    * var out = 1.0 / (1.0 + exp(-input)); inputGrad = outputGrad * out * (1.0 - out)
    * ```
    *
    * @returns the output tensor
    */
    sigmoid(): Tensor {
        return ops.sigmoid(this);
    }
    /**
    * ![Plot of sigmoid and its gradient](/plots/sigmoid.svg)
    *
    * Calculates:
    * ```js
    * output = 1.0 / (1.0 + exp(-input))
    * ```
    *
    * Gradient:
    * ```js
    * var out = 1.0 / (1.0 + exp(-input)); inputGrad = outputGrad * out * (1.0 - out)
    * ```
    *
    * @returns the output tensor
    */
    sigmoid_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sigmoid_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of sign and its gradient](/plots/sign.svg)
    *
    * Calculates:
    * ```js
    * output = sign(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    sign(): Tensor {
        return ops.sign(this);
    }
    /**
    * ![Plot of sign and its gradient](/plots/sign.svg)
    *
    * Calculates:
    * ```js
    * output = sign(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    sign_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sign_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of silu and its gradient](/plots/silu.svg)
    *
    * Calculates:
    * ```js
    * output = input / (1.0 + exp(-input))
    * ```
    *
    * Gradient:
    * ```js
    * var out = 1.0 / (1.0 + exp(-input)); inputGrad = outputGrad * (out + input * out * (1.0 - out))
    * ```
    *
    * @returns the output tensor
    */
    silu(): Tensor {
        return ops.silu(this);
    }
    /**
    * ![Plot of silu and its gradient](/plots/silu.svg)
    *
    * Calculates:
    * ```js
    * output = input / (1.0 + exp(-input))
    * ```
    *
    * Gradient:
    * ```js
    * var out = 1.0 / (1.0 + exp(-input)); inputGrad = outputGrad * (out + input * out * (1.0 - out))
    * ```
    *
    * @returns the output tensor
    */
    silu_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("silu_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of sin and its gradient](/plots/sin.svg)
    *
    * Calculates:
    * ```js
    * output = sin(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * cos(input)
    * ```
    *
    * @returns the output tensor
    */
    sin(): Tensor {
        return ops.sin(this);
    }
    /**
    * ![Plot of sin and its gradient](/plots/sin.svg)
    *
    * Calculates:
    * ```js
    * output = sin(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * cos(input)
    * ```
    *
    * @returns the output tensor
    */
    sin_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sin_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of sinc and its gradient](/plots/sinc.svg)
    *
    * Calculates:
    * ```js
    * var inpi = input * 3.141592653589793; output = input == 0.0 ? 1.0 : sin(inpi) / inpi
    * ```
    *
    * Gradient:
    * ```js
    * var inpi = input * 3.141592653589793; inputGrad = input == 0.0 ? 0.0 : (outputGrad * 3.141592653589793 * (inpi*cos(inpi) - sin(inpi)) / (inpi*inpi))
    * ```
    *
    * @returns the output tensor
    */
    sinc(): Tensor {
        return ops.sinc(this);
    }
    /**
    * ![Plot of sinc and its gradient](/plots/sinc.svg)
    *
    * Calculates:
    * ```js
    * var inpi = input * 3.141592653589793; output = input == 0.0 ? 1.0 : sin(inpi) / inpi
    * ```
    *
    * Gradient:
    * ```js
    * var inpi = input * 3.141592653589793; inputGrad = input == 0.0 ? 0.0 : (outputGrad * 3.141592653589793 * (inpi*cos(inpi) - sin(inpi)) / (inpi*inpi))
    * ```
    *
    * @returns the output tensor
    */
    sinc_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sinc_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of sinh and its gradient](/plots/sinh.svg)
    *
    * Calculates:
    * ```js
    * output = sinh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * cosh(input)
    * ```
    *
    * @returns the output tensor
    */
    sinh(): Tensor {
        return ops.sinh(this);
    }
    /**
    * ![Plot of sinh and its gradient](/plots/sinh.svg)
    *
    * Calculates:
    * ```js
    * output = sinh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * cosh(input)
    * ```
    *
    * @returns the output tensor
    */
    sinh_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sinh_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of sqrt and its gradient](/plots/sqrt.svg)
    *
    * Calculates:
    * ```js
    * output = sqrt(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (2.0 * sqrt(input))
    * ```
    *
    * @returns the output tensor
    */
    sqrt(): Tensor {
        return ops.sqrt(this);
    }
    /**
    * ![Plot of sqrt and its gradient](/plots/sqrt.svg)
    *
    * Calculates:
    * ```js
    * output = sqrt(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (2.0 * sqrt(input))
    * ```
    *
    * @returns the output tensor
    */
    sqrt_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("sqrt_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of square and its gradient](/plots/square.svg)
    *
    * Calculates:
    * ```js
    * output = input * input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * 2.0 * input
    * ```
    *
    * @returns the output tensor
    */
    square(): Tensor {
        return ops.square(this);
    }
    /**
    * ![Plot of square and its gradient](/plots/square.svg)
    *
    * Calculates:
    * ```js
    * output = input * input
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * 2.0 * input
    * ```
    *
    * @returns the output tensor
    */
    square_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("square_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = input - other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad; otherGrad = -outputGrad
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @param alpha the alpha value to multiply `other` with
    * @returns the output tensor
    */
    sub(other: Tensor, alpha?: number): Tensor {
        return ops.sub(this, other, alpha);
    }
    /**
    * Alias for `sub`.
    *
    * Calculates:
    * ```js
    * output = input - other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad; otherGrad = -outputGrad
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @param alpha the alpha value to multiply `other` with
    * @returns the output tensor
    */
    subtract(other: Tensor, alpha?: number): Tensor {
        return ops.sub(this, other, alpha);
    }
    /**
    * Calculates:
    * ```js
    * output = input - other
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad; otherGrad = -outputGrad
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @param alpha the alpha value to multiply `other` with
    * @returns the output tensor
    */
    sub_(other: Tensor, alpha?: number): Tensor {
        const params = {
            size: shapeSize(this.shape),
            alpha: alpha || 1.0,
        };
        return this.runKernelInplace("sub_", { dtype: this.dtype }, params, other);
    }
    /**
    * ![Plot of tan and its gradient](/plots/tan.svg)
    *
    * Calculates:
    * ```js
    * output = tan(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (cos(input) * cos(input))
    * ```
    *
    * @returns the output tensor
    */
    tan(): Tensor {
        return ops.tan(this);
    }
    /**
    * ![Plot of tan and its gradient](/plots/tan.svg)
    *
    * Calculates:
    * ```js
    * output = tan(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / (cos(input) * cos(input))
    * ```
    *
    * @returns the output tensor
    */
    tan_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("tan_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of tanh and its gradient](/plots/tanh.svg)
    *
    * Calculates:
    * ```js
    * output = tanh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * (1.0 - tanh(input) * tanh(input))
    * ```
    *
    * @returns the output tensor
    */
    tanh(): Tensor {
        return ops.tanh(this);
    }
    /**
    * ![Plot of tanh and its gradient](/plots/tanh.svg)
    *
    * Calculates:
    * ```js
    * output = tanh(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * (1.0 - tanh(input) * tanh(input))
    * ```
    *
    * @returns the output tensor
    */
    tanh_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("tanh_", { dtype: this.dtype }, params);
    }
    /**
    * ![Plot of trunc and its gradient](/plots/trunc.svg)
    *
    * Calculates:
    * ```js
    * output = trunc(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    trunc(): Tensor {
        return ops.trunc(this);
    }
    /**
    * Alias for `trunc`.
    *
    * ![Plot of trunc and its gradient](/plots/trunc.svg)
    *
    * Calculates:
    * ```js
    * output = trunc(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    fix(): Tensor {
        return ops.trunc(this);
    }
    /**
    * ![Plot of trunc and its gradient](/plots/trunc.svg)
    *
    * Calculates:
    * ```js
    * output = trunc(input)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = 0
    * ```
    *
    * @returns the output tensor
    */
    trunc_(): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("trunc_", { dtype: this.dtype }, params);
    }
    /**
    * Calculates:
    * ```js
    * output = input == 0.0 ? 0.0 : input * log(other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = input == 0.0 ? 0.0 : outputGrad * log(other); otherGrad = input == 0.0 ? 0.0 : outputGrad * (input / other);
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    xlogy(other: Tensor): Tensor {
        return ops.xlogy(this, other);
    }
    /**
    * Calculates:
    * ```js
    * output = input == 0.0 ? 0.0 : input * log(other)
    * ```
    *
    * Gradient:
    * ```js
    * inputGrad = input == 0.0 ? 0.0 : outputGrad * log(other); otherGrad = input == 0.0 ? 0.0 : outputGrad * (input / other);
    * ```
    *
    * @param other the other tensor whose shape is broadcastable with the input tensor
    * @returns the output tensor
    */
    xlogy_(other: Tensor): Tensor {
        const params = {
            size: shapeSize(this.shape),
        };
        return this.runKernelInplace("xlogy_", { dtype: this.dtype }, params, other);
    }
    /**
    * Calculates:
    * ```js
    * output = output && input
    * ```
    *
    * with an initial value of `output = 1`.
    *
    * Gradient:
    * ```js
    * inputGrad = output ? outputGrad : 0.0
    * ```
    *
    * @returns the output tensor
    */
    all(dim?: number, keepdim?: boolean): Tensor {
        return ops.all(this);
    }
    /**
    * Calculates:
    * ```js
    * output = output || input
    * ```
    *
    * with an initial value of `output = 0`.
    *
    * Gradient:
    * ```js
    * inputGrad = output ? outputGrad : 0.0
    * ```
    *
    * @returns the output tensor
    */
    any(dim?: number, keepdim?: boolean): Tensor {
        return ops.any(this);
    }
    /**
    * Calculates:
    * ```js
    * output = output + input
    * ```
    *
    * with an initial value of `output = 0.0`.
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad / inputSize
    * ```
    *
    * @returns the output tensor
    */
    mean(dim?: number, keepdim?: boolean): Tensor {
        return ops.mean(this);
    }
    /**
    * Calculates:
    * ```js
    * output = output + input * input
    * ```
    *
    * with an initial value of `output = 0.0`.
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * input / output
    * ```
    *
    * @returns the output tensor
    */
    norm(dim?: number, keepdim?: boolean): Tensor {
        return ops.norm(this);
    }
    /**
    * Calculates:
    * ```js
    * output = output * input
    * ```
    *
    * with an initial value of `output = 1.0`.
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad * output / input
    * ```
    *
    * @returns the output tensor
    */
    prod(dim?: number, keepdim?: boolean): Tensor {
        return ops.prod(this);
    }
    /**
    * Calculates:
    * ```js
    * output = output + input
    * ```
    *
    * with an initial value of `output = 0.0`.
    *
    * Gradient:
    * ```js
    * inputGrad = outputGrad
    * ```
    *
    * @returns the output tensor
    */
    sum(dim?: number, keepdim?: boolean): Tensor {
        return ops.sum(this);
    }
    /**
    * Calculates:
    * ```js
    * output = output + (input != 0)
    * ```
    *
    * with an initial value of `output = 0.0`.
    *
    * @returns the output tensor
    */
    countNonzero(dim?: number, keepdim?: boolean): Tensor {
        return ops.countNonzero(this);
    }
    // End codegen marker
}
