import { ATypedArray, Dtype } from "./dtype";
import { Shape, defaultStrides, shapeSize } from "./shape";
import { TensorArrayData } from "./tensor_if";

export class StorageBase {
}

export abstract class UntypedStorage extends StorageBase {
    abstract get buffer(): ArrayBuffer | null;
}

export class ArrayBufferStorage extends UntypedStorage {
    private _buffer: ArrayBuffer;
    constructor(byteSize: number) {
        super();
        this._buffer = new ArrayBuffer(byteSize);
    }
    get buffer(): ArrayBuffer | null {
        return this._buffer;
    }
}

export class GPUBufferStorage extends UntypedStorage {
    private _buffer: ArrayBuffer | null = null;
    constructor(byteSize: number, alignment: number) {
        super();
    }
    get buffer(): ArrayBuffer | null {
        return this._buffer;
    }
}

export function newTypedArrayForDtype(length: number, dtype: Dtype) {
    switch (dtype) {
        case "uint8":
            return new Uint8Array(length);
        case "int32":
            return new Int32Array(length);
        case "float32":
            return new Float32Array(length);
        default:
            throw new Error(`Unsupported dtype: ${dtype}`);
    }
}

export function newTypedArrayFromArray(data: TensorArrayData | null, dtype: Dtype, allocFor: (shape: Shape) => UntypedStorage): {data: UntypedStorage, shape: number[], strides: number[]} {
    const shape: number[] = [];
    function getShape(data: TensorArrayData | number) {
        if (typeof data === "number") {
            return;
        }
        shape.push(data.length);
        getShape(data[0]);
    }
    if (data !== null) {
        getShape(data);
    }
    const strides = defaultStrides(shape);
    const size = shapeSize(shape);
    const flatData = new Float32Array(size);
    let flatIndex = 0;
    function flatten(data: TensorArrayData | number) {
        if (typeof data === "number") {
            flatData[flatIndex] = data;
            flatIndex++;
            return;
        }
        for (let i = 0; i < data.length; i++) {
            let d = data[i];
            if (typeof d === "number") {
                for (let j = 0; j < data.length; j++) {
                    flatData[flatIndex] = data[j] as number;
                    flatIndex++;
                }
                return;
            }
            flatten(data[i]);
        }
    }
    if (data !== null) {
        flatten(data);
    }
    return {data: flatData, shape, strides};
}
