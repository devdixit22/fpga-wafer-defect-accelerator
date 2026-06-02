`timescale 1ns/1ps

module system_tb;
    parameter N = 8;
    
    // We pass these from iverilog command line, but define defaults here
    parameter IN_BLOCKS = 3;
    parameter OUT_BLOCKS = 1;
    parameter NUM_PATCHES = 256;

    reg clk;
    reg reset;
    reg start;
    wire done;

    system_top #(
        .N(N),
        .IN_BLOCKS(IN_BLOCKS),
        .OUT_BLOCKS(OUT_BLOCKS),
        .NUM_PATCHES(NUM_PATCHES)
    ) DUT (
        .clk(clk),
        .reset(reset),
        .start(start),
        .done(done)
    );

    always #5 clk = ~clk;

    integer fd, i, j;

    initial begin
        clk = 0;
        reset = 1;
        start = 0;

        #20 reset = 0;
        #10 start = 1;
        #10 start = 0;

        begin : wait_done
            integer timeout;
            timeout = 0;
            while (!done && timeout < 5000000) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            if (timeout >= 5000000) begin
                $display("TIMEOUT waiting for done");
                $finish;
            end
        end

        // Dump output memory to output.mem
        // Each output word contains N x 32-bit values packed into (N*32)-bit words
        fd = $fopen("output.mem", "w");
        for (i = 0; i < OUT_BLOCKS * NUM_PATCHES; i = i + 1) begin
            for (j = 0; j < N; j = j + 1) begin
                $fdisplay(fd, "%0d", $signed(DUT.out_buf.memory[i][((j+1)*32)-1 -: 32]));
            end
        end
        $fclose(fd);

        $display("Simulation complete.");
        #20 $finish;
    end
endmodule
