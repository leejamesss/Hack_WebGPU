import { CodeWriter } from "../src/opgen";
import { OpSpec, ReductionOpSpec } from "../src/op_spec";
import { opKernelSpecs } from "../src/kernels_opgen";
import { registry as opRegistry } from "../src/op_table";

// import fs
import * as fs from "fs";

console.log("Running code generator...");

const absSrcDir = fs.realpathSync(__dirname + "/../src");
console.log("src dir:", absSrcDir);

function insertCodegenIntoFile(path: string, codegen: string): void {
    const code = fs.readFileSync(path, "utf8");
    const codegenMarker = "// Codegen marker";
    const endCodegenMarker = "// End codegen marker";
    const startIndex = code.indexOf(codegenMarker);
    const endIndex = code.indexOf(endCodegenMarker);
    if (startIndex === -1 || endIndex === -1) {
        throw new Error("Could not find codegen marker in " + path);
    }
    const pre = code.slice(0, startIndex + codegenMarker.length);
    const post = code.slice(endIndex);
    const newCode = pre + "\n" + codegen + "\n    " + post;
    console.log("Writing", path);
    fs.writeFileSync(path, newCode);
}

function writeOpHeader(opSpec: OpSpec, name: string, isAlias: boolean, suffix: string, w: CodeWriter) {
    const hasAlpha = opSpec.alpha ?? false;
    const isBinary = opSpec.type === "binary";
    const isReduction = opSpec.type === "reduction";
    writeOpDocs(opSpec, "this", isAlias, w);
    if (isReduction) {
        w.writeLine(`${name}(dim?: number, keepdim?: boolean): Tensor${suffix}`);
    }
    else if (isBinary) {
        if (hasAlpha) {
            w.writeLine(`${name}(other: Tensor, alpha?: number): Tensor${suffix}`);
        }
        else {
            w.writeLine(`${name}(other: Tensor): Tensor${suffix}`);
        }
    }
    else {
        if (hasAlpha) {
            w.writeLine(`${name}(alpha?: number): Tensor${suffix}`);
        }
        else {
            w.writeLine(`${name}(): Tensor${suffix}`);
        }
    }
};

// Write the Tensor class
function writeTensorCode(): void {
    const w = new CodeWriter();
    w.indent();
    for (const [opSpec, kernelSpec] of opKernelSpecs) {
        const isInplace = kernelSpec.name.endsWith("_");
        const isGrad = kernelSpec.name.endsWith("Grad");
        if (isGrad) continue;
        const isBinary = opSpec.type === "binary";
        const hasAlpha = opSpec.alpha ?? false;
        const isReduction = opSpec.type === "reduction";        
        writeOpHeader(opSpec, kernelSpec.name, false, " {", w);
        w.indent();
        if (isInplace) {
            if (isBinary) {
                w.writeLine(`const params = {`);
                w.indent();
                w.writeLine(`size: shapeSize(this.shape),`);
                if (hasAlpha) {
                    w.writeLine(`alpha: alpha || 1.0,`);
                }
                w.dedent();
                w.writeLine(`};`);
                w.writeLine(`return this.runKernelInplace("${kernelSpec.name}", { dtype: this.dtype }, params, other);`);
            }
            else {
                w.writeLine(`const params = {`);
                w.indent();
                w.writeLine(`size: shapeSize(this.shape),`);
                if (hasAlpha) {
                    w.writeLine(`alpha: alpha || 1.0,`);
                }
                w.dedent();
                w.writeLine(`};`);
                w.writeLine(`return this.runKernelInplace("${kernelSpec.name}", { dtype: this.dtype }, params);`);
            }
        }
        else {
            if (isBinary) {
                if (hasAlpha) {
                    if (isInplace) {
                        w.writeLine(`this._impl.${kernelSpec.name}(other._impl, alpha);`);
                        w.writeLine(`return this;`);
                    }
                    else {
                        w.writeLine(`return ops.${kernelSpec.name}(this, other, alpha);`);
                    }
                }
                else {
                    if (isInplace) {
                        w.writeLine(`this._impl.${kernelSpec.name}(other._impl);`);
                        w.writeLine(`return this;`);
                    }
                    else {
                        w.writeLine(`return ops.${kernelSpec.name}(this, other);`);
                    }
                }
            }
            else {
                if (hasAlpha) {
                    if (isInplace) {
                        w.writeLine(`this._impl.${kernelSpec.name}(alpha);`);
                        w.writeLine(`return this;`);
                    }
                    else {
                        w.writeLine(`return ops.${kernelSpec.name}(this, alpha);`);
                    }
                }
                else {
                    if (isInplace) {
                        w.writeLine(`this._impl.${kernelSpec.name}();`);
                        w.writeLine(`return this;`);
                    }
                    else {
                        w.writeLine(`return ops.${kernelSpec.name}(this);`);
                    }
                }
            }
        }
        w.dedent();
        w.writeLine(`}`);
        if (!isInplace) {
            for (const alias of opSpec.aliases ?? []) {
                writeOpHeader(opSpec, alias, true, " {", w);
                w.indent();
                if (isBinary) {
                    if (hasAlpha) {
                        w.writeLine(`return ops.${kernelSpec.name}(this, other, alpha);`);
                    }
                    else {
                        w.writeLine(`return ops.${kernelSpec.name}(this, other);`);
                    }
                }
                else {
                    if (hasAlpha) {
                        w.writeLine(`return ops.${kernelSpec.name}(this, alpha);`);
                    }
                    else {
                        w.writeLine(`return ops.${kernelSpec.name}(this);`);
                    }
                }
                w.dedent();
                w.writeLine(`}`);
            }
        }
    }
    const code = w.toString();
    // console.log(code);
    const path = absSrcDir + "/tensor.ts";
    insertCodegenIntoFile(path, code);
}
writeTensorCode();

// Write the Tensor class
function writeTensorDeclCode(): void {
    const w = new CodeWriter();
    w.indent();
    for (const [opSpec, kernelSpec] of opKernelSpecs) {
        const isInplace = kernelSpec.name.endsWith("_");
        const isGrad = kernelSpec.name.endsWith("Grad");
        if (isGrad) continue;
        writeOpHeader(opSpec, kernelSpec.name, false, ";", w);
        if (!isInplace) {
            for (const alias of opSpec.aliases ?? []) {
                writeOpHeader(opSpec, alias, true, ";", w);
            }
        }
    }
    const code = w.toString();
    // console.log(code);
    const path = absSrcDir + "/tensor.d.ts";
    // insertCodegenIntoFile(path, code);
}
writeTensorDeclCode();

// Write autograd functions
function writeFunctionsCode(): void {
    const w = new CodeWriter();
    w.writeLine(`import {
    AutoFunction,
    FunctionInput,
    GradientContext,
    GradientFunctionOutput,
} from "./autograd";
import type { Tensor } from "./tensor";
import { shapeSize } from "./shape";`);
    for (const [opSpec, kernelSpec] of opKernelSpecs) {
        const isInplace = kernelSpec.name.endsWith("_");
        if (isInplace) {
            continue;
        }
        const isGrad = kernelSpec.name.endsWith("Grad");
        if (isGrad) continue;
        const isBinary = opSpec.type === "binary";
        const isReduction = opSpec.type === "reduction";
        const hasAlpha = opSpec.alpha ?? false;
        const className = kernelSpec.name[0].toUpperCase() + kernelSpec.name.slice(1) + "Function";
        const config: {[name: string]: number|string} = {dtype: "float32"};
        let outputShapesS: string = "[input.shape]";
        if (isReduction) {
            outputShapesS = "[[]]";
            config["workgroupSize"] = 64;
        }
        const configS = JSON.stringify(config);
        const writeUnpackInputs = (inputsName: string, includeAlpha: boolean) => {
            if (isReduction) {
                w.writeLine(`const [input] = ${inputsName} as [Tensor];`);
            }
            else if (isBinary) {
                if (hasAlpha && includeAlpha) {
                    w.writeLine(`const [input, other, alpha] = ${inputsName} as [Tensor, Tensor, number|undefined];`);
                }
                else {
                    w.writeLine(`const [input, other] = ${inputsName} as [Tensor, Tensor];`);
                }
            }
            else {
                if (hasAlpha && includeAlpha) {
                    w.writeLine(`const [input, alpha] = ${inputsName} as [Tensor, number|undefined];`);
                }
                else {
                    w.writeLine(`const [input] = ${inputsName} as [Tensor];`);
                }
            }
        }
        const writeParams = (alphaName: string) => {
            w.writeLine(`const params = {`);
            w.indent();
            w.writeLine(`size: shapeSize(input.shape),`);
            if (hasAlpha) {
                w.writeLine(`alpha: ${alphaName} || 1.0,`);
            }
            w.dedent();
            w.writeLine(`};`);
        };
        w.writeLine(`export class ${className} extends AutoFunction {`);
        w.indent();

        // Forward
        w.writeLine(`static forward(inputs: FunctionInput[]): Tensor {`);
        w.indent();
        writeUnpackInputs("inputs", true);
        writeParams("alpha");
        w.writeLine(`if (!input.isContiguous) { throw new Error("Input must be contiguous"); }`);
        if (isBinary) {
            w.writeLine(`if (!other.isContiguous) { throw new Error("Other must be contiguous"); }`);
            w.writeLine(`return input.runKernel("${kernelSpec.name}", ${configS}, params, ${outputShapesS}, other)[0];`);
        }
        else {
            w.writeLine(`return input.runKernel("${kernelSpec.name}", ${configS}, params, ${outputShapesS})[0];`);
        }
        w.dedent();
        w.writeLine(`}`);

        w.writeLine(`static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ): void {`);
        w.indent();
        writeUnpackInputs("inputs", true);
        if (isBinary) {
            if (hasAlpha) {
                w.writeLine(`ctx.alpha = alpha;`);
            }
            w.writeLine(`ctx.saveForBackward(input, other);`);
        }
        else {
            if (hasAlpha) {
                w.writeLine(`ctx.alpha = alpha;`);
            }
            w.writeLine(`ctx.saveForBackward(input);`);
        }
        w.dedent();
        w.writeLine(`}`);

        // Backward
        w.writeLine(`static backward(ctx: GradientContext, outputGrad: Tensor): GradientFunctionOutput[] {`);
        w.indent();
        writeUnpackInputs("ctx.savedTensors", false);
        writeParams("ctx.alpha");
        if (isReduction) {
            w.writeLine(`return input.runKernel("${kernelSpec.name}Grad", ${configS}, params, [input.shape], outputGrad);`);
        }
        else if (isBinary) {
            w.writeLine(`return input.runKernel("${kernelSpec.name}Grad", ${configS}, params, [input.shape, other.shape], other, outputGrad);`);
        }
        else {
            w.writeLine(`return input.runKernel("${kernelSpec.name}Grad", ${configS}, params, [input.shape], outputGrad);`);
        }
        w.dedent();
        w.writeLine(`}`);
        w.dedent();
        w.writeLine(`}`);
    }
    w.writeLine("");
    const code = w.toString();
    // console.log(code);
    const path = absSrcDir + "/functions_opgen.ts";
    console.log("Writing", path);
    fs.writeFileSync(path, code);
}
writeFunctionsCode();

function writeOpDocs(opSpec: OpSpec, inputName: string, isAlias: boolean, w: CodeWriter, includeParamsAndReturn: boolean = true): void {
    const isBinary = opSpec.type === "binary";
    const isUnary = opSpec.type === "unary";
    const hasAlpha = opSpec.alpha ?? false;
    w.writeLine(`/**`);
    if (isAlias) {
        w.writeLine(`* Alias for \`${opSpec.name}\`.`);
        w.writeLine(`*`);
    }
    if (isUnary) {
        w.writeLine(`* ![Plot of ${opSpec.name} and its gradient](/plots/${opSpec.name}.svg)`);
        w.writeLine(`*`);
    }
    w.writeLine(`* Calculates:`);
    w.writeLine(`* \`\`\`js`);
    w.writeLine(`* ${opSpec.forward}`);
    w.writeLine(`* \`\`\``);
    w.writeLine(`*`);
    if (opSpec.type === "reduction") {
        w.writeLine(`* with an initial value of \`${(opSpec as ReductionOpSpec).init}\`.`);
        w.writeLine(`*`);
    }
    if (opSpec.backward) {
        w.writeLine(`* Gradient:`);
        w.writeLine(`* \`\`\`js`);
        w.writeLine(`* ${opSpec.backward}`);
        w.writeLine(`* \`\`\``);
        w.writeLine(`*`);
    }
    if (includeParamsAndReturn) {
        if (inputName !== "this") {
            w.writeLine(`* @param ${inputName} the input tensor of any shape`);
        }
        if (isBinary) {
            w.writeLine(`* @param other the other tensor whose shape is broadcastable with the input tensor`);
            if (hasAlpha) {
                w.writeLine(`* @param alpha the alpha value to multiply \`other\` with`);
            }
            else {
            }
        }
        else {
            if (hasAlpha) {
                w.writeLine(`* @param alpha the alpha value`);
            }
            else {
            }
        }
        w.writeLine(`* @returns the output tensor`);
    }
    w.writeLine(`*/`);
}

// Write global ops
function writeOpsCode(): void {
    const w = new CodeWriter();
    w.writeLine(`import * as functions from "./functions_opgen";
import { Tensor } from "./tensor";
import { unary, unaryWithAlpha, binary, binaryWithAlpha } from "./ops_high";`);
    for (const [opSpec, kernelSpec] of opKernelSpecs) {
        const isInplace = kernelSpec.name.endsWith("_");
        if (isInplace) {
            continue;
        }
        const isGrad = kernelSpec.name.endsWith("Grad");
        if (isGrad) continue;
        const isBinary = opSpec.type === "binary";
        const hasAlpha = opSpec.alpha ?? false;
        const funcName = kernelSpec.name[0].toUpperCase() + kernelSpec.name.slice(1) + "Function";
        const writeHeader = (name: string, isAlias: boolean) => {
            writeOpDocs(opSpec, "input", isAlias, w);
            if (isBinary) {
                if (hasAlpha) {
                    w.writeLine(`export function ${name}(input: Tensor, other: Tensor, alpha?: number): Tensor {`);
                }
                else {
                    w.writeLine(`export function ${name}(input: Tensor, other: Tensor): Tensor {`);
                }
            }
            else {
                if (hasAlpha) {
                    w.writeLine(`export function ${name}(input: Tensor, alpha?: number): Tensor {`);
                }
                else {
                    w.writeLine(`export function ${name}(input: Tensor): Tensor {`);
                }
            }
        };
        writeHeader(kernelSpec.name, false);
        w.indent();
        if (isBinary) {
            if (hasAlpha) {
                w.writeLine(`return binaryWithAlpha(functions.${funcName}, input, other, alpha);`);
            }
            else {
                w.writeLine(`return binary(functions.${funcName}, input, other);`);
            }
        }
        else {
            if (hasAlpha) {
                w.writeLine(`return unaryWithAlpha(functions.${funcName}, input, alpha);`);
            }
            else {
                w.writeLine(`return unary(functions.${funcName}, input);`);
            }
        }
        w.dedent();
        w.writeLine(`}`);
        for (const alias of opSpec.aliases ?? []) {
            writeHeader(alias, true);
            w.indent();
            if (isBinary) {
                if (hasAlpha) {
                    w.writeLine(`return ${kernelSpec.name}(input, other, alpha);`);
                }
                else {
                    w.writeLine(`return ${kernelSpec.name}(input, other);`);
                }
            }
            else {
                if (hasAlpha) {
                    w.writeLine(`return ${kernelSpec.name}(input, alpha);`);
                }
                else {
                    w.writeLine(`return ${kernelSpec.name}(input);`);
                }
            }
            w.dedent();
            w.writeLine(`}`);
        }
    }
    w.writeLine("");
    const code = w.toString();
    // console.log(code);
    const path = absSrcDir + "/ops_opgen.ts";
    console.log("Writing", path);
    fs.writeFileSync(path, code);
}
writeOpsCode();

// Write autograd functions
function writeNNModulesCode(): void {
    const w = new CodeWriter();
    w.writeLine(`import { Tensor } from "./tensor";
import { Module } from "./nn_module";`);
    for (const opSpec of opRegistry) {
        if (!opSpec.nnOp)
            continue;
        const name = opSpec.name;
        const isBinary = opSpec.type === "binary";
        const isReduction = opSpec.type === "reduction";
        const hasAlpha = opSpec.alpha ?? false;
        const nnName = opSpec.nnName ?? (name[0].toUpperCase() + name.slice(1));
        const params: string[] = ["input: Tensor"];
        const args: string[] = [];
        if (isBinary) {
            params.push("other: Tensor");
            args.push("other");
        }
        if (hasAlpha) {
            params.push("alpha?: number");
            args.push("alpha");
        }
        const paramsStr = params.join(", ");
        const argsStr = args.join(", ");
        writeOpDocs(opSpec, "input", false, w, false);
        w.writeLine(`export class ${nnName} extends Module {`);
        w.indent();

        // Forward
        w.writeLine(`forward(${paramsStr}): Tensor {`);
        w.indent();
        w.writeLine(`return input.${opSpec.name}(${argsStr});`);
        w.dedent();
        w.writeLine(`}`);

        // End Module
        w.dedent();
        w.writeLine(`}`);
    }
    w.writeLine("");
    const code = w.toString();
    // console.log(code);
    const path = absSrcDir + "/nn_opgen.ts";
    console.log("Writing", path);
    fs.writeFileSync(path, code);
}
writeNNModulesCode();

