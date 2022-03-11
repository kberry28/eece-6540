__kernel void pi_calc( int floats_per_item, __local float local_result, 
     __global float* global_result) {

   /* initialize local summation */
   /*so this local variable is visible to all kernels running on the device, right?*/
   /*in that case do we need an array of as many floats as total items/work items local?*/
   local_result = 0;

   /* Make sure previous processing has completed */
   /**is this necessary now?**/
   barrier(CLK_LOCAL_MEM_FENCE);

   //use for calculating fractions in each work item
   int index = get_global_id(0);

   /* Iterate through characters in text */
   for(int i=1; i<floats_per_item; i++) {


      /* Check for 'that' */
      if(all(check_vector.s0123))
         atomic_inc(local_result);

      /* Check for 'with' */
      if(all(check_vector.s4567))
         atomic_inc(local_result + 1);

      /* Check for 'have' */
      if(all(check_vector.s89AB))
         atomic_inc(local_result + 2);

      /* Check for 'from' */
      if(all(check_vector.sCDEF))
         atomic_inc(local_result + 3);
   }

   /* Make sure local processing has completed */
   barrier(CLK_GLOBAL_MEM_FENCE);

   /* Perform global reduction */
   if(get_local_id(0) == 0) {
      atomic_add(global_result, local_result[0]);
      atomic_add(global_result + 1, local_result[1]);
      atomic_add(global_result + 2, local_result[2]);
      atomic_add(global_result + 3, local_result[3]);
   }
}