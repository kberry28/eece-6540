__kernel void pi_calc(int floats_per_item, int localint, __local float* local_result, 
     __global float* global_result) {

   /*global ID is work item from entire input set*/
   size_t index = get_global_id(0);
   size_t group = get_local_id(0);
   printf("k\n");

   /* add-sub up 4 consecutive fractions and save value in local mem array*/
   /*location in local mem array is local_id*/
   /*value to calculate fraction terms is global_id*/
   for(int i=0; i<floats_per_item; i++) {
   	if(i%2)
   		//if i is odd, subtract
   		local_result[group]-= 1/(8*index+i*2+1);
   	else
   		//if i is even, add
   		local_result[group]+= 1/(8*index+i*2+1);
   }

   /* Make sure local processing for the work group has completed */

   barrier(CLK_GLOBAL_MEM_FENCE);

   /* Perform global reduction */
   /*first work item now sums the local memory array and puts it in its "bucket" in global array*/
   if(group == 0) {
      int bucket = index/localint;
   	for (int c=0; c<localint;c++)
      /*this global_result is the global memory array buffer ON THE DEVICE*/
      global_result[bucket]+=local_result[c];
   }
}