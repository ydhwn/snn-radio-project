// -----------------------------------------------------------------------------
// Module: snn_top.v
// Description: Top-level SNN Architecture for Modulation Classification on FPGA
// -----------------------------------------------------------------------------

module snn_top #(
    parameter INPUT_DIM = 512,
    parameter HIDDEN_DIM = 64,
    parameter NUM_CLASSES = 6,
    parameter DATA_WIDTH = 16
)(
    input  wire                   clk,
    input  wire                   reset,
    input  wire [DATA_WIDTH-1:0]  iq_in,      // Serial IQ data from UART/ADC
    input  wire                   data_valid,
    output wire [NUM_CLASSES-1:0] class_spikes, // Parallel output spikes
    output reg                    busy
);

    // Internal wires for spike propagation
    wire [INPUT_DIM-1:0]  input_spikes;
    wire [HIDDEN_DIM-1:0] hidden_spikes;

    // 1. INPUT ENCODER (Temporal Spike Generator)
    // Converts IQ values to spike trains based on magnitude
    spike_encoder #(DATA_WIDTH) encoder_inst (
        .clk(clk),
        .reset(reset),
        .data_in(iq_in),
        .valid(data_valid),
        .spikes_out(input_spikes)
    );

    // 2. HIDDEN LAYER (64 LIF Neurons)
    // Processes the input spike patterns
    genvar i;
    generate
        for (i = 0; i < HIDDEN_DIM; i = i + 1) begin : hidden_layer
            lif_neuron #(DATA_WIDTH) neuron_h (
                .clk(clk),
                .reset(reset),
                .synaptic_input(input_spikes[i]), // Simplified 1-to-1 for RTL demo
                .spike_out(hidden_spikes[i])
            );
        end
    endgenerate

    // 3. OUTPUT LAYER (6 LIF Neurons - one per Modulation)
    generate
        for (i = 0; i < NUM_CLASSES; i = i + 1) begin : output_layer
            lif_neuron #(DATA_WIDTH) neuron_o (
                .clk(clk),
                .reset(reset),
                .synaptic_input(|hidden_spikes), // OR-reduction for demo
                .spike_out(class_spikes[i])
            );
        end
    endgenerate

endmodule
