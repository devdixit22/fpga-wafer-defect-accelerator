`timescale 1ns/1ps

// FIX: cycle_count now resets after completion so repeated inferences work.
//      compute_done is a 1-cycle pulse, not a sticky flag.
module array_controller #(
    parameter N = 4
)(
    input clk,
    input reset,
    input start,

    output reg load_weights,
    output reg stream_activations,
    output reg compute_done
);

// States
localparam IDLE     = 2'd0;
localparam LOADING  = 2'd1;
localparam RUNNING  = 2'd2;
localparam DONE     = 2'd3;

reg [1:0]  state;
reg [15:0] cycle_count;

always @(posedge clk) begin
    if (reset) begin
        state             <= IDLE;
        cycle_count       <= 0;
        load_weights      <= 0;
        stream_activations<= 0;
        compute_done      <= 0;
    end else begin
        // Default: de-assert pulse signals
        compute_done <= 0;

        case (state)
            IDLE: begin
                load_weights       <= 0;
                stream_activations <= 0;
                cycle_count        <= 0;
                if (start)
                    state <= LOADING;
            end

            LOADING: begin
                load_weights       <= 1;
                stream_activations <= 0;
                cycle_count        <= cycle_count + 1;
                if (cycle_count == N - 1) begin
                    cycle_count <= 0;
                    state       <= RUNNING;
                end
            end

            RUNNING: begin
                load_weights       <= 0;
                stream_activations <= 1;
                cycle_count        <= cycle_count + 1;
                // FIX: wait N + N - 1 cycles for pipeline to drain
                if (cycle_count == (2*N - 1)) begin
                    cycle_count <= 0;
                    state       <= DONE;
                end
            end

            DONE: begin
                stream_activations <= 0;
                compute_done       <= 1;  // 1-cycle pulse
                state              <= IDLE;
            end
        endcase
    end
end

endmodule
