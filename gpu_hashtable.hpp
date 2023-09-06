#ifndef _HASHCPU_
#define _HASHCPU_

/*
    @name   Dumitrescu Alexandra
    @date   June 2023
    @for    ACS - Parellel Hashtable

*/

/*

|  struct containing entry values in the hashmap

*/
typedef struct hashtable_entry hashtable_entry;
struct hashtable_entry {
        int32_t key;
        int32_t value;
};


/*

|  struct containing relevant data that will be sent to the kernel methods

*/
typedef struct device_hashmap_t device_hashmap_t;
struct device_hashmap_t {
       int *maximum_buckets;
       unsigned int *current_buckets;
       hashtable_entry *hashmap;
};


/*

|  class containing the hashtable

*/
class GpuHashTable
{
        public:
                GpuHashTable(int size);

                void reshape(int sizeReshape);
                bool insertBatch(int *keys, int* values, int numKeys);
                int* getBatch(int* key, int numItems);
                
                ~GpuHashTable();

        protected:
                /* circular array of hashtable entries */
                hashtable_entry *hashmap;
                
                /* number of current set buckets */
                int32_t curr_buckets;

                /* maximum number of avilable buckets */
                int32_t max_buckets;

                /* hash function */
                unsigned int (*hash_function)(unsigned int);

                /* maximum number of threads per block */
                int maximum_number_of_threads_per_block;

        private:
                /* depending on the CUDA device, computes the maximum number of threads per block */
                int get_number_of_threads_per_block();

                /* checks for each insertion if the load factor is more than 0.75f */
                bool more_than_load_factor(int N);

                /* computes the number of needed blocks to complete the given number of tasks */
                int get_number_of_needed_blocks(int tasks, int max);

                /* copies keys and values from host to device */
                void copy_entries_host_device(int *host_keys, int *host_values, int **device_keys, int **device_values, int num_keys);

                /* copies the number of maximum blocks and current set blocks from host to device */
                void copy_buckets_info_host_device(int curr_buckets, int max_buckets, unsigned int **device_curr_buckets, int **device_max_buckets);

                /* fills the structure of info on device hashmap with the given params */
                void compute_hashmap_device(device_hashmap_t **device_hashmap, int *max_buckets, unsigned int *curr_buckets, hashtable_entry *hashmap);
               

};

#endif
