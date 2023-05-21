import type { Tensor } from "./tensor";
import { TensorBase } from "./tensor_base";

export type FunctionInput = Tensor | number | boolean | string | undefined;
export type GradientFunctionOutput = Tensor | null;

export function isTensor(input: FunctionInput): input is Tensor {
    return input instanceof TensorBase;
}

export class GradientContext {
    [key: string]: any;
    needsInputGradient: boolean[];
    inputsWithGradient: (Tensor | null)[];
    savedTensors: Tensor[] = [];
    constructor(inputs: FunctionInput[]) {
        this.needsInputGradient = inputs.map(
            (input) => isTensor(input) && input.requiresGrad
        );
        this.inputsWithGradient = inputs.map((input) =>
            isTensor(input) && input.requiresGrad ? input : null
        );
    }
    saveForBackward(...tensors: Tensor[]) {
        this.savedTensors = tensors;
    }
}

export type GradientFunction = (
    ctx: GradientContext,
    output: Tensor
) => (Tensor | null)[];

export interface IAutoFunction {
    forward(inputs: FunctionInput[]): Tensor; // function to use without grad
    apply(...inputs: FunctionInput[]): Tensor; // function to use when requiring grad
    backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[];
}

export class AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        throw new Error("Do not call forward on AutoFunction directly.");
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        throw new Error("Do not call setupContext on AutoFunction directly.");
    }
    static backward(
        ctx: GradientContext,
        outputGrad: Tensor
    ): GradientFunctionOutput[] {
        throw new Error("Do not call backward on AutoFunction directly.");
    }
    static apply(...inputs: FunctionInput[]): Tensor {
        const ctx = new GradientContext(inputs);
        const detachedInputs = inputs.map((input) =>
            isTensor(input) ? input.detach() : input
        );
        const output = this.forward(detachedInputs);
        this.setupContext(ctx, detachedInputs, output);
        output.setGradientFunction(ctx, this.backward);
        return output;
    }
}

export class ArtisanalFunction {
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {
        const tensorInputs = inputs as Tensor[];
        ctx.saveForBackward(...tensorInputs);
    }
    static backward(
        ctx: GradientContext,
        outputGrad: Tensor
    ): GradientFunctionOutput[] {
        throw new Error("Do not call backward on AutoFunction directly.");
    }

    // This function only records the tensors for backward, DON'T run forward in this function, you should pass the output tensor from forward result
    static applyGrad(allInputs: FunctionInput[],output:Tensor): Tensor {
        const ctx = new GradientContext(allInputs);
        const detachedInputs = allInputs.map((input) =>
            isTensor(input) ? input.detach() : input
        );
        this.setupContext(ctx, detachedInputs, output);
        output.setGradientFunction(ctx, this.backward);
        return output;
    }
}

export function shouldCreateGradient(...inputs: Tensor[]): boolean {
    for (const input of inputs) {
        if (input.requiresGrad) {
            return true;
        }
    }
    return false;
}
