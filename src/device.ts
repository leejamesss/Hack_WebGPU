import { Shape, Shapeish } from "./shape";
import { Dtype } from "./dtype";
import { TensorArrayData, TensorImpl } from "./tensor_if";

export type DeviceType = "cpu" | "webgpu";
export type DeviceId = string;

export type Deviceish = DeviceType | Device | DeviceId;

export abstract class Device {
    private _id: DeviceId;
    private _type: DeviceType;
    get id(): DeviceId {
        return this._id;
    }
    get type(): DeviceType {
        return this._type;
    }
    constructor(id: DeviceId, type: DeviceType) {
        this._id = id;
        this._type = type;
    }
    abstract ones(shape: Shape, dtype: Dtype): TensorImpl;
    abstract tensor(data: TensorArrayData, dtype: Dtype): TensorImpl;
    abstract zeros(shape: Shape, dtype: Dtype): TensorImpl;
}