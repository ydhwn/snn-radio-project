// -----------------------------------------------------------------------------
// Module: spike_encoder.v
// Description: Converts 16-bit fixed-point IQ data into spiking trains
// -----------------------------------------------------------------------------

module spike_encoder #(
    parameter DATA_WIDTH = 16
)(
    input  wire                   clk,
    input  wire                   reset,
    input  wire [DATA_WIDTH-1:0]  data_in,
    input  wire                   valid,
    output reg  [511:0]           spikes_out
);

    integer i;
    reg [DATA_WIDTH-1:0] threshold_array [511:0];

    // Simple Rate Encoding: If magnitude > threshold, spike.
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            spikes_out <= 0;
            // Initialize random thresholds for rate encoding simulation
            for (i = 0; i < 512; i = i + 1) begin
                threshold_array[i] <= i * (16'hFFFF / 512);
            end
        end else if (valid) begin
            for (i = 0; i < 512; i = i + 1) begin
                // If data magnitude is greater than a distributed threshold, fire.
                if (data_in > threshold_array[i])
                    spikes_out[i] <= 1;
                else
                    spikes_out[i] <= 0;
            end
        end else begin
            spikes_out <= 0;
        end
    end

endmodule
