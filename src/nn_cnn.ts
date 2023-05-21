import { Module } from "./nn_module";
import {LinearFunction} from './functions_artisanal';
import {Tensor} from './tensor';
import {createFromSize as arrayFromSize} from './utils';
import { conv2d } from "./ops_artisanal";
import { Dtype } from "./dtype";
export class AvgPooling2d extends Module {}

export class Conv2d extends Module {
    inChannels: number;
    outChannels: number;
    kernelSize: number | [number, number];
    stride: number | [number, number];
    padding: number | [number, number] | "valid" | "same";
    dtype: Dtype;
    weight: Tensor;
    // TODO: bias
    constructor(
        inChannels: number,
        outChannels: number,
        kernelSize: number | [number, number],
        stride: number | [number, number],
        padding: number | [number, number] | "valid" | "same",
        dtype: Dtype
    ) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.dtype = dtype;
        if (kernelSize instanceof Array){
            this.weight = new Tensor(arrayFromSize([outChannels,inChannels,...kernelSize]),dtype=dtype)
        }else{
            this.weight = new Tensor(arrayFromSize([outChannels,inChannels,kernelSize,kernelSize]))
        }

    }
        forward(input: Tensor): Tensor {
        return conv2d(input, this.weight, undefined, this.stride, this.padding)
    }    

}

export class ConvTranspose2d extends Module {}

export class GroupNorm extends Module {
    numGroups: number;
    numChannels: number;
    constructor(numGroups: number, numChannels: number) {
        super();
        this.numGroups = numGroups;
        this.numChannels = numChannels;
    }
}
export class Linear extends Module {
    inChannels: number;
    outChannels: number;
    weight: Tensor;
    bias: Tensor;

    constructor(inChannels: number, outChannels: number) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        
        this.weight = new Tensor(arrayFromSize([outChannels,inChannels]));
        this.bias = new Tensor(new Array([outChannels]));
    }
    forward(input: Tensor): Tensor {
        return LinearFunction.apply(input, this.weight, this.bias);
    }
}