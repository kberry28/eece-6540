/*kernel is given the pointer on-device memory where its summed result will go*/
/*each work item will put its result in its local index*/

__kernel void pi_calc(int floats_per_item, int local_size, __local float* local_result, 
     __global float* global_result) {

   //global ID is work item from entire input set
   int index = get_global_id(0);
   int local = get_local_id(0);

   /* add-sub up 4 consecutive fractions and save value in local mem array*/
   /*location in local mem array is local_id*/
   /*value to calculate fraction terms is global_id*/
   for(int i=0; i<floats_per_item; i++) {
   	if(i%2)
   		//if i is odd, subtract
   		local_result[local]-= 1/(8*index+i*2+1);
   	else
   		//if i is even, add
   		local_result[local]+= 1/(8*index+i*2+1);
   }

   /* Make sure local processing for the work group has completed */

   barrier(CLK_GLOBAL_MEM_FENCE);

   /* Perform global reduction */
   /*first work item now sums the local memory array and puts it in its "bucket" in global array*/
   if(local == 0) {
      int bucket = index/local_size;
   	for (int c=0, c<local_size,c++)
      /*this global_result is the global memory array buffer ON THE DEVICE*/
      global_result[bucket]+=local_result[c]);
   }
}