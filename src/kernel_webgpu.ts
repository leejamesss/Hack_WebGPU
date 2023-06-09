import { Device } from "./device";
import { EvalEnv } from "./expr";
import {
    Kernel,
    KernelConfig,
    KernelInputSpec,
    KernelOutputSpec,
    KernelParamsInput,
    KernelSpec,
    getKernelShaderCode,
    getShaderTypeElementByteSize,
} from "./kernel";

export class KernelWebGPU extends Kernel {
    private _gpuDevice: GPUDevice;
    private _bindGroupLayout: GPUBindGroupLayout;
    private _computePipeline: GPUComputePipeline;
    private _shaderCode: string;

    constructor(spec: KernelSpec, config: KernelConfig, device: Device) {
        super(spec, config, device);
        const gpuDevice = (device as any).gpuDevice;
        if (!gpuDevice) {
            throw new Error("Cannot create a GPU kernel without a GPU device");
        }
        this._gpuDevice = gpuDevice;
        let bindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = [];
        let bindingIndex = 0;
        for (let i = 0; i < spec.inputs.length; i++, bindingIndex++) {
            bindGroupLayoutEntries.push({
                binding: bindingIndex,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage" as GPUBufferBindingType,
                },
            });
        }
        for (let i = 0; i < spec.outputs.length; i++, bindingIndex++) {
            bindGroupLayoutEntries.push({
                binding: bindingIndex,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage" as GPUBufferBindingType,
                },
            });
        }
        bindGroupLayoutEntries.push({
            binding: bindingIndex,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: "read-only-storage" as GPUBufferBindingType,
            },
        });
        this._bindGroupLayout = gpuDevice.createBindGroupLayout({
            entries: bindGroupLayoutEntries,
        });
        this._shaderCode = getKernelShaderCode(spec, config);
        const shaderModule = gpuDevice.createShaderModule({
            code: this._shaderCode,
        });
        this._computePipeline = gpuDevice.createComputePipeline({
            layout: gpuDevice.createPipelineLayout({
                bindGroupLayouts: [this._bindGroupLayout],
            }),
            compute: {
                module: shaderModule,
                entryPoint: "main",
            },
        });
    }
    run(
        inputs: GPUBuffer[],
        parameters: KernelParamsInput,
        outputs?: GPUBuffer[]
    ): GPUBuffer[] {
        // console.log("run gpu kernel", this.key);

        // Build the parameter environment
        const env: EvalEnv = this.getRunEnv(parameters);
        let paramsBufferSize = 0;
        for (let i = 0; i < this.spec.parameters.length; i++) {
            const param = this.spec.parameters[i];
            paramsBufferSize += getShaderTypeElementByteSize(param.shaderType);
        }

        // Get input buffers with storage usage
        const storageInputs = this.spec.inputs.map((input, i) =>
            this.getStorageInputBuffer(
                input,
                inputs[i] ? inputs[i] : null,
                i,
                env
            )
        );

        // Get output buffers with storage usage
        const storageOutputs = this.spec.outputs.map((output, i) =>
            this.getStorageOutputBuffer(
                output,
                outputs ? outputs[i] : null,
                i,
                env
            )
        );

        // Build the params buffer
        const paramsBuffer = this._gpuDevice.createBuffer({
            mappedAtCreation: true,
            size: paramsBufferSize,
            usage: GPUBufferUsage.STORAGE,
        });
        const paramsArrayBuffer = paramsBuffer.getMappedRange();
        for (let paramDtype of ["u32", "f32"]) {
            let paramsArray = new (
                paramDtype === "u32" ? Uint32Array : Float32Array
            )(paramsArrayBuffer);
            for (let i = 0; i < this.spec.parameters.length; i++) {
                const param = this.spec.parameters[i];
                if (param.shaderType === paramDtype) {
                    paramsArray[i] = +env[param.name];
                }
            }
        }
        paramsBuffer.unmap();

        // Bind the buffers
        const bindGroup = this.createBindGroup(
            storageInputs,
            paramsBuffer,
            storageOutputs
        );

        // Get the workgroup counts
        const [workgroupCountX, workgroupCountY, workgroupCountZ] =
            this.getWorkgroupCounts(env);

        // Start a new command encoder
        const commandEncoder = this._gpuDevice.createCommandEncoder();

        // Encode the kernel using pass encoder
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this._computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(
            workgroupCountX,
            workgroupCountY,
            workgroupCountZ
        );
        passEncoder.end();

        // Submit GPU commands
        const gpuCommands = commandEncoder.finish();
        this._gpuDevice.queue.submit([gpuCommands]);

        // Return the storage output buffers
        return storageOutputs;
    }
    private getStorageInputBuffer(
        inputSpec: KernelInputSpec,
        providedInput: GPUBuffer | null,
        inputIndex: number,
        env: EvalEnv
    ): GPUBuffer {
        if (providedInput === null) {
            throw new Error(
                `Missing input buffer #${inputIndex} (out of ${this.spec.inputs.length}) named "${inputSpec.name}" in kernel "${this.key}"`
            );
        }
        if (providedInput.usage & GPUBufferUsage.STORAGE) {
            providedInput.unmap();
            return providedInput;
        } else {
            throw new Error("Provided input buffer is not a storage buffer");
        }
    }
    private getStorageOutputBuffer(
        outputSpec: KernelOutputSpec,
        providedOutput: GPUBuffer | null,
        outputIndex: number,
        env: EvalEnv
    ): GPUBuffer {
        if (providedOutput !== null) {
            if (providedOutput.usage & GPUBufferUsage.STORAGE) {
                providedOutput.unmap();
                return providedOutput;
            } else {
                throw new Error(
                    "Provided output buffer is not a storage buffer"
                );
            }
        } else {
            const outputElementByteSize = getShaderTypeElementByteSize(
                outputSpec.shaderType
            );
            const outputElementCount = Math.ceil(
                this._outputSizeFuncs[outputIndex](env)
            );
            // console.log("output size", outputElementCount, outputElementByteSize);
            const outputBufferSize = outputElementByteSize * outputElementCount;
            const outputBuffer = this._gpuDevice.createBuffer({
                mappedAtCreation: false,
                size: outputBufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });
            return outputBuffer;
        }
    }
    private createBindGroup(
        inputBuffers: GPUBuffer[],
        paramsBuffer: GPUBuffer,
        outputBuffers: GPUBuffer[]
    ): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [];
        let bindingIndex = 0;
        for (let i = 0; i < inputBuffers.length; i++, bindingIndex++) {
            entries.push({
                binding: bindingIndex,
                resource: {
                    buffer: inputBuffers[i],
                },
            });
        }
        for (let i = 0; i < this.spec.outputs.length; i++, bindingIndex++) {
            const outputBuffer = outputBuffers[i];
            entries.push({
                binding: bindingIndex,
                resource: {
                    buffer: outputBuffer,
                },
            });
        }
        entries.push({
            binding: bindingIndex,
            resource: {
                buffer: paramsBuffer,
            },
        });
        const bindGroup = this._gpuDevice.createBindGroup({
            layout: this._bindGroupLayout,
            entries: entries,
        });
        return bindGroup;
    }
}
