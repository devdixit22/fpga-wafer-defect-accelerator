`timescale 1ns/1ps

module accelerator_tb;

parameter N = 4;

reg clk;
reg reset;
reg start;

// Flat packed buses
reg [(N*N*8)-1:0]  weights;
reg [(N*8)-1:0]    activations;
reg [(N*32)-1:0]   bias;

wire [(N*32)-1:0]  outputs;
wire compute_done;

accelerator_top #(.N(N)) DUT (
    .clk         (clk),
    .reset       (reset),
    .start       (start),
    .weights     (weights),
    .activations (activations),
    .bias        (bias),
    .outputs     (outputs),
    .compute_done(compute_done)
);

always #5 clk = ~clk;

integer i, j, fd;

initial begin
    clk         = 0;
    reset       = 1;
    start       = 0;
    weights     = 0;
    activations = 0;
    bias        = 0;

    @(posedge clk); @(posedge clk);
    reset = 0;

    // Identity weights: weights[i][i] = 1, rest = 0
    // activations[i] = 1 for all i
    // Expected output[i] = 1 for all i (identity * ones = ones)
    for (i = 0; i < N; i = i + 1) begin
        // Pack activation i into bits [(i+1)*8-1 : i*8]
        activations[((i+1)*8)-1 -: 8] = 8'sd1;
        // Pack bias i = 0
        bias[((i+1)*32)-1 -: 32] = 32'sd0;
        // Pack diagonal weight = 1
        weights[((i*N + i + 1)*8)-1 -: 8] = 8'sd1;
    end

    @(posedge clk);
    start = 1;

    // Wait for done pulse with timeout
    begin : wait_done
        integer timeout;
        timeout = 0;
        while (!compute_done && timeout < 1000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        if (timeout >= 1000) begin
            $display("TIMEOUT waiting for compute_done");
            $finish;
        end
    end

    @(posedge clk);
    start = 0;

    fd = $fopen("output.mem", "w");
    for (i = 0; i < N; i = i + 1) begin
        $fdisplay(fd, "%0d", $signed(outputs[((i+1)*32)-1 -: 32]));
        $display("output[%0d] = %0d", i, $signed(outputs[((i+1)*32)-1 -: 32]));
    end
    $fclose(fd);

    $display("Simulation complete.");
    #20 $finish;
end

initial begin
    #500000;
    $display("HARD TIMEOUT");
    $finish;
end

endmodule
