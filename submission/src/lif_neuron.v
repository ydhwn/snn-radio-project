// -----------------------------------------------------------------------------
// Module: lif_neuron.v
// Project: Energy-Efficient SNN Modulation Classifier (B.Tech Major Project)
// -----------------------------------------------------------------------------
// This module implements a single Leaky Integrate-and-Fire (LIF) neuron in 
// RTL (Verilog). It is the fundamental hardware block of our Spiking Neural Network.
// -----------------------------------------------------------------------------

module lif_neuron #(
    parameter DATA_WIDTH = 16,
    parameter THRESHOLD = 16'h1000, // 4096 in fixed-point
    parameter LEAK_FACTOR = 16'h0F00 // 0.93 leakage (approx)
)(
    input  wire                   clk,
    input  wire                   reset,
    input  wire [DATA_WIDTH-1:0]  synaptic_input, // Weighted sum from prev layer
    output reg                    spike_out,
    output reg  [DATA_WIDTH-1:0]  membrane_potential
);

    reg [DATA_WIDTH-1:0] v_next;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            membrane_potential <= 0;
            spike_out <= 0;
        end else begin
            // 1. Check for spike threshold
            if (membrane_potential >= THRESHOLD) begin
                spike_out <= 1;
                membrane_potential <= 0; // Reset after firing
            end else begin
                spike_out <= 0;
                // 2. Integration with Leakage: v[t+1] = (v[t] * leak) + input
                // Note: Simplified fixed-point multiplication for FPGA
                v_next = ( (membrane_potential * LEAK_FACTOR) >> 12 ) + synaptic_input;
                membrane_potential <= v_next;
            end
        end
    end

endmodule
