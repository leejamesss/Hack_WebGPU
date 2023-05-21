import { Module } from "./nn_module";
import {LinearFunction} from './functions_artisanal';
import {Tensor} from './tensor';
import {createFromSize as arrayFromSize} from './utils';
import { Conv2dFunction } from "./ops_artisanal";
import { createFromSizeUniform} from './utils'
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
            this.weight = new Tensor(createFromSizeUniform([outChannels,inChannels,...kernelSize]),dtype=dtype)
        }else{
            this.weight = new Tensor(createFromSizeUniform([outChannels,inChannels,kernelSize,kernelSize]))
        }
        this.weight.requiresGrad = true

    }
        forward(input: Tensor): Tensor {
        return Conv2dFunction.apply(input, this.weight, undefined, this.stride, this.padding)
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

    constructor(inChannels: number, outChannels: number, dtype:Dtype = "float32") {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        
        this.weight = new Tensor(createFromSizeUniform([outChannels,inChannels]),dtype);
        this.bias = new Tensor(createFromSizeUniform([outChannels]),dtype);

        this.weight.requiresGrad = true
        this.bias.requiresGrad = true
    }
    forward(input: Tensor): Tensor {
        return LinearFunction.apply(input, this.weight, this.bias);
    }

    BP(delta:number = 0.005){
        if(!this.weight.grad){
            throw new Error('weight not found')
        }
        if(!this.bias.grad){
            throw new Error('bias not found')
        }
        this.weight = this.weight.add(this.weight.grad,delta)
        console.log(this.bias.grad)
        // this.bias = this.bias.add(this.bias.grad,delta)
        this.weight.grad = null
        this.bias.grad = null
        
    }
}

