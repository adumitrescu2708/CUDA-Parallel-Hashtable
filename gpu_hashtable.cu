
#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"

using namespace std;
#define ERROR -1
#define LOAD_FACTOR 0.7f


/*
    @name   Dumitrescu Alexandra
    @date   June 2023
    @for    ACS - Parellel Hashtable

*/

/*

| hash function()
| https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key/12996028#12996028
|
| Marking the function with the keyword __device__ to mark down that the function should be in the device
| memory and that it is accessible and executable on the device (GPU).

*/
static __device__ unsigned int hash_int(unsigned int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}


/*

| get_number_of_threads_per_block()
|
| Assuming all CUDA device have the same properties, this method is used for retriving the number of maximum
| number of threads that could be launched within a single CUDA block on the given device.
| After extracting the needed information, check for possible errors and print possible error message for
| debugg purposes.
|

*/
int GpuHashTable::get_number_of_threads_per_block()
{
        int maximum_number_of_threads_per_block = 0;
        cudaError_t err_attr = cudaDeviceGetAttribute(&maximum_number_of_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
        if(err_attr != cudaSuccess) {
			const char *err_message = cudaGetErrorString(err_attr);
			cout << err_message;
           	return ERROR;
        }
        return maximum_number_of_threads_per_block;
}

/*

| kernel_ht_get()
| @param1 - array of given keys, stored on the GPU
| @param2 - array of given values, stored on the GPU
| @param3 - number of elements
| @param4 - struct containing relevant information on hashmap, stored on GPU
|
| The main idea of the parallel get function is to make each thread inspect one element in the
| keys array and store the found value in the values array. Given the possible number of threads that
| is not equal to the number of elements, we must restrict those threads. This way, only the first
| *NUM_KEYS* threads will search for the corresponding key.
| When searching for the value, we first apply the hash function on the corresponding key and
| start searching for it circularly in the buffer of elements of the hashmap, since we are using
| LINEAR PROABING for treating the possible collisions.
| The trip count value is meant to count how many times we have itterated through the circular buffer.
| Since only one thread will update a certain element in the buffer of values, there is no need for
| atomic operations on the update of the array.
|

*/
__global__ void kernel_ht_get(int32_t *key, int32_t *values, int N, device_hashmap_t *device_hashmap) {
 unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
 if(i < N) {
    if(device_hashmap == NULL) {
        return;
    }
     int max = (*(device_hashmap->maximum_buckets));
     hashtable_entry *hashmap = (device_hashmap->hashmap);


     int serched_key = key[i];
     int position = hash_int(serched_key) % (max);
     bool found = false;
     int trip_count = 0;
     for(int e = position; ; e++) {
         hashtable_entry entry = hashmap[e];
         if(entry.key == serched_key) {
                 values[i] = entry.value;
                 found = true;
         }
         if(found) {
                 return;
         }
         if(e == max - 1) {
                 e = 0;
         }
         if(e == position) {
                 trip_count ++;
         }
         if(trip_count == 2) {
                 return;
         }
     }
 }
}

/*

| kernel_ht_reshape()
| @param1 - struct containing relevant information on hashmap, stored on GPU
| @param2 - new hashmap
| @param3 - new number of maximum buckets
|
| This method follows the same idea previously stated above at the get() function. What differs is that
| when trying to update the hashtable we need atomic operations and for this I chose atomicCAS.
| Each yhread is meant to take one element from the previous array and rehash it in the new array.
| If the previous value of the key is 0, meaning no element was added (note. initially the hashmap is 0 - set),
| there is no need for a remapping.
| We compute the new index by applying the hash function on the corresponding key and iterate using
| LINEAR PROABING in the circular buffer until we find an empty slot, if available.

*/
__global__ void kernel_ht_reshape(device_hashmap_t *device_hashmap, hashtable_entry *new_hashtable, int new_max_buckets)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(device_hashmap == NULL) {
        return;
    }
    int max_buckets = *(device_hashmap->maximum_buckets);
    if(i < max_buckets) {
        hashtable_entry *hashtable = device_hashmap->hashmap;

        int new_position = hash_int(hashtable[i].key) % new_max_buckets;
        bool inserted = false;
        int trip_count = 0;
        if(hashtable[i].key == 0) {
            return;
        }
        for(int e = new_position; ; e++) {
            int old = atomicCAS(&new_hashtable[e].key, 0, hashtable[i].key);
            inserted = old == 0 ? true : false;
            if(inserted) {
                new_hashtable[e].value = hashtable[i].value;
                return;
            }

            if(e == new_position) {
                trip_count ++;
            }

            if(e == new_max_buckets - 1) {
                e = 0;
            }

            if(trip_count == 2) {
                return;
            }
        }
    }
}


/*

| kernel_ht_insert()
| @param1 - array of given keys, stored on the GPU
| @param2 - array of given values, stored on the GPU
| @param3 - number of elements
| @param4 - struct containing relevant information on hashmap, stored on GPU
|
| This method works similarly to the previous methods. What differs is that we need
| to update the number of current busy buckets in the hashmap. For updating this
| integer shared by all blocks, we need atomic operations. What is more,
| the given keys might correspond to previously stored elements, in which case we
| updte the value and do not increment the value. (note. we use LINEAR PROABING
| for solving colisions)
|

*/
__global__ void kernel_ht_insert(int *key, int *value, int N, device_hashmap_t *device_hashmap)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(device_hashmap == NULL) {
        return;
    }
    if(i < N) {
        int max_buckets = *(device_hashmap->maximum_buckets);
        hashtable_entry *hashtable = device_hashmap->hashmap;

        int hash_index = hash_int(key[i]) % (max_buckets);
        bool inserted = false;
        int trip_count = 0;
        int e = hash_index;
        for(; ; e++) {
            int old = atomicCAS(&hashtable[e].key, 0, key[i]);
            inserted = old == 0 || old == key[i] ? true : false;
            if(inserted && old == 0) {
                atomicAdd(device_hashmap->current_buckets, 1);
            }
            if(inserted) {
                hashtable[e].value = value[i];
                return;
            }
            if(e == (max_buckets) - 1) {
                e = 0;
            }
            if(e == hash_index) {
                trip_count ++;
            }
            if(trip_count == 2) {
                return;
            }
        }
    }
}


/*

| get_number_of_needed_blocks()
| @param1 - the number of tasks
| @param2 - maximum number of threads per GPU block
|
| This method is used for retrieving the number of needed blocks
| for completing the given number of tasks. We compute the ceil(number_of_tasks/max_threads_per_block)
|

*/
int GpuHashTable::get_number_of_needed_blocks(int tasks, int max) {
    if(tasks < 0) {
        return 0;
    }
    int blocks_number = tasks / max;
    if(blocks_number * max < tasks) {
        blocks_number += 1;
    }
    return blocks_number;
}

/*

| copy_entries_host_device()
| @param1 - keys, stored on CPU
| @param2 - values, stored on CPU
| @param3 - address of keys array, stored on GPU
| @param4 - address of values array, stored on GPU
| @param5 - number of elements
|
| This method is used for copying the keys and values from CPU to GPU in order to send them to global functions
| executed by each thread in the GPU block. This method also prints possible errors to the stdout for debugg purpose
| In case of get method, @param2 will be NULL set. For this case we set the device array with 0.

*/
void GpuHashTable::copy_entries_host_device(int *host_keys, int *host_values, int **device_keys, int **device_values, int num_keys)
{
	cudaError_t err;
    const char *err_message;
    
    err = glbGpuAllocator->_cudaMalloc((void **)device_keys, num_keys * sizeof(int));
    if(err != cudaSuccess) {
        err_message = cudaGetErrorString(err);
        cout << err_message << endl;
        return;
    }

    err = glbGpuAllocator->_cudaMalloc((void **)device_values, num_keys * sizeof(int));
    if(err != cudaSuccess) {
        err_message = cudaGetErrorString(err);
        cout << err_message << endl;
        return;
    }

	cudaMemcpy(*device_keys, host_keys, num_keys * sizeof(int), cudaMemcpyHostToDevice);
	if(host_values == NULL) {
		cudaMemset(*device_values, 0, num_keys * sizeof(int));
	} else {
		cudaMemcpy(*device_values, host_values, num_keys * sizeof(int), cudaMemcpyHostToDevice);
	}
}


/*

| copy_entries_host_device()
| @param1 - keys, stored on CPU
| @param2 - values, stored on CPU
| @param3 - address of keys array, stored on GPU
| @param4 - address of values array, stored on GPU
| @param5 - number of elements
|
| This method is used for copying the keys and values from CPU to GPU in order to send them to global functions
| executed by each thread in the GPU block. This method also prints possible errors to the stdout for debugg purpose
| In case of get method, @param2 will be NULL set. For this case we set the device array with 0.

*/
void GpuHashTable::copy_buckets_info_host_device(int curr_buckets, int max_buckets, unsigned int **device_curr_buckets, int **device_max_buckets)
{
    cudaError_t malloc_curr;
    if(device_curr_buckets != NULL) {
        malloc_curr = glbGpuAllocator->_cudaMallocManaged((void **)device_curr_buckets, sizeof(unsigned int));
    } else {
        malloc_curr = cudaSuccess;
    }
	
	cudaError_t malloc_max = glbGpuAllocator->_cudaMallocManaged((void **)device_max_buckets, sizeof(unsigned int));

	if(malloc_curr != cudaSuccess || malloc_max != cudaSuccess) {
        if(malloc_curr != cudaSuccess) {
            const char *err_message = cudaGetErrorString(malloc_curr);
            cout << err_message;
            return;
        }
        if(malloc_max != cudaSuccess) {
            const char *err_message = cudaGetErrorString(malloc_max);
            cout << err_message;
            return;
        }
	}
    if(device_curr_buckets != NULL) {
        (*(*device_curr_buckets)) = curr_buckets;
    }
    
    (*(*device_max_buckets)) = max_buckets;

}


/*

| compute_hashmap_device()
| @param1 - pointer to new device_hashmap, stored on GPU
| @param2 - pointer to maximum available buckets in hashmap, stored on GPU
| @param3 - hashmap, stored on GPU
|
| This method computes the additional struct that will be sent to the kernel methods, containing relevant
| info on the hashmap. It allocates memory, checks for possible errors and fills the attributes.

*/
void GpuHashTable::compute_hashmap_device(device_hashmap_t **device_hashmap, int *max_buckets, unsigned int *curr_buckets, hashtable_entry *hashmap)
{
    cudaError_t err = glbGpuAllocator->_cudaMallocManaged((void **)device_hashmap, sizeof(device_hashmap_t));

    if(err != cudaSuccess) {
        const char *err_message = cudaGetErrorString(err);
        cout << err_message << endl;
        return;
    }

    (*device_hashmap)->maximum_buckets = max_buckets;
    if(curr_buckets != NULL) {
        (*device_hashmap)->current_buckets = curr_buckets;
    }
    (*device_hashmap)->hashmap = hashmap;

}


/*

| GpuHashTable() - CONSTRUCTOR
|
| Allocates memory for the circular buffer of hashtable entries and 0-sets it, checks for possible
| errors and prints error log to stdout. The initial number of maximum available buckets will be
| the given size and the current set buckets will be set to 0. We also find the number
| of maximum threads per GPU block and store it.

*/
GpuHashTable::GpuHashTable(int size) {
    int32_t len = size * sizeof(hashtable_entry);
    cudaError_t err;
    const char *err_message = NULL;

    err = glbGpuAllocator->_cudaMallocManaged((void **) &hashmap, len);
    if(err != cudaSuccess) {
        err_message = cudaGetErrorString(err);
        cout << err_message << endl;
        return;
    }

    err = cudaMemset(hashmap, 0, len);
    if(err != cudaSuccess) {
        err_message = cudaGetErrorString(err);
        cout << err_message << endl;
        return;
    }

    max_buckets = size;
    curr_buckets = 0;
    maximum_number_of_threads_per_block = get_number_of_threads_per_block();
    hash_function = &hash_int;
}


/*

| GpuHashTable() - DECONSTRUCTOR
|
| Frees the memory allocated for the circular buffer of hashtable entries.

*/
GpuHashTable::~GpuHashTable() {
    glbGpuAllocator->_cudaFree(hashmap);
}

/*

| reshape()
|
| Steps followed by the reshape function
| A) allocate memory on GPU for the new hashtable, set it to 0 and check for possible errors
| B) get the number of needed blocks for <max_buckets> rehashing tasks
| C) copy the needed info from host to device
| D) reshape, free memory
| E) update the hashtable and the maximum buckets parameters 

*/
void GpuHashTable::reshape(int numBucketsReshape) {
    hashtable_entry *resized_hashmap;
    cudaError_t err;
    const char *err_message;

    /* (A) */
    err = glbGpuAllocator->_cudaMallocManaged((void **)&resized_hashmap, numBucketsReshape * sizeof(hashtable_entry));
    if(err != cudaSuccess) {
        err_message = cudaGetErrorString(err);
        cout << err_message << endl;
        return;
    }

    int len = numBucketsReshape * sizeof(hashtable_entry);
    err = cudaMemset(resized_hashmap, 0, len);
    if(err != cudaSuccess) {
        err_message = cudaGetErrorString(err);
        cout << err_message << endl;
        return;
    }

    /* (B) */
    int blocks_number = get_number_of_needed_blocks(max_buckets, maximum_number_of_threads_per_block);

    /* (C) */
    int *maximum_buckets;
    copy_buckets_info_host_device(curr_buckets, max_buckets, NULL, &maximum_buckets);
    device_hashmap_t *device_hashmap;
    compute_hashmap_device(&device_hashmap, maximum_buckets, NULL, hashmap);

    kernel_ht_reshape<<<blocks_number, maximum_number_of_threads_per_block>>>(device_hashmap, resized_hashmap, numBucketsReshape);
	err = cudaDeviceSynchronize();

    /* (D) */
    if(err != cudaSuccess) {
        err_message = cudaGetErrorString(err);
        cout << err_message;
        return;
    }

    glbGpuAllocator->_cudaFree(hashmap);
    glbGpuAllocator->_cudaFree(maximum_buckets);
    glbGpuAllocator->_cudaFree(device_hashmap);

    /* (E) */
    hashmap = resized_hashmap;
    max_buckets = numBucketsReshape;
}


/*

| more_than_load_factor()
| @param1 - number of elements that will be inserted in the hashtable
|
| Computes the load factor and checks wether it is more than 0.75
|

*/
bool GpuHashTable::more_than_load_factor(int N) {
        double curr_load_factor = (double) (curr_buckets + N) / (max_buckets);
        return curr_load_factor > 0.75f;
}


/*

| insertBatch()
|
| Steps followed by the getBatch function
| A) get the number of needed blocks for <numKeys> insertion tasks
| B) allocate memory on GPU for the device keys and values, copies the data from the host and check for possible errors
| C) checks the load factor and if needed reshapes with 2X dimension
| D) insert, free memory
| E) updates the number of set buckets in the hashtable

*/
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
    if(keys == NULL || values == NULL || numKeys < 0) {
        return false;
    }

    /* (A) */
    int blocks_number = get_number_of_needed_blocks(numKeys, maximum_number_of_threads_per_block);
    int *device_keys, *device_values, *device_max_buckets;
    unsigned int *updated_current_buckets;

    /* (B) */
    copy_entries_host_device(keys, values, &device_keys, &device_values, numKeys);

    /* (C) */
    if(more_than_load_factor(numKeys)) {
        if((max_buckets) != 0)
            reshape((double) 2 * (max_buckets));
        else
            reshape((double) 2 * (numKeys));
    }

    copy_buckets_info_host_device(curr_buckets, max_buckets, &updated_current_buckets, &device_max_buckets);

    device_hashmap_t *device_hashmap;
    compute_hashmap_device(&device_hashmap, device_max_buckets, updated_current_buckets, hashmap);

    /* (D) */
    kernel_ht_insert<<<blocks_number, maximum_number_of_threads_per_block>>>(device_keys, device_values, numKeys, device_hashmap);
    cudaError_t err_sync = cudaDeviceSynchronize();

    if(err_sync != cudaSuccess) {
        const char *err_message = cudaGetErrorString(err_sync);
        cout << err_message;
        return false;
    }

    /* (E) */
    curr_buckets = (*updated_current_buckets);

    glbGpuAllocator->_cudaFree(device_keys);
    glbGpuAllocator->_cudaFree(device_values);
    glbGpuAllocator->_cudaFree(updated_current_buckets);
    glbGpuAllocator->_cudaFree(device_max_buckets);
    glbGpuAllocator->_cudaFree(device_hashmap);
    
    return true;
}


/*

| getBatch()
|
| Steps followed by the getBatch function
| A) allocate memory on GPU for the device keys and values, copies the data from the host and check for possible errors
| B) get the number of needed blocks for <numKeys> get tasks
| C) copy the needed info from host to device
| D) get, free memory
| E) copy the info on the host array of values

*/
int* GpuHashTable::getBatch(int* keys, int numKeys) {
    int *values = (int *) malloc(numKeys * sizeof(int));
    int *device_values, *device_keys, *maximum_buckets;

    cudaError_t err;
    const char *err_message = NULL;

    /* (A) */
    err = glbGpuAllocator->_cudaMallocManaged((void **)&device_values, numKeys * sizeof(int));
    if(err != cudaSuccess) {
        err_message = cudaGetErrorString(err);
        cout << err_message << endl;
        return NULL;
    }

    err = glbGpuAllocator->_cudaMalloc((void **)&device_keys, numKeys * sizeof(int));
    if(err != cudaSuccess) {
        err_message = cudaGetErrorString(err);
        cout << err_message << endl;
        return NULL;
    }

    err = cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    if(err != cudaSuccess) {
        err_message = cudaGetErrorString(err);
        cout << err_message << endl;
        return NULL;
    }

    err = cudaMemset(device_values, 0, numKeys * sizeof(int));
    if(err != cudaSuccess) {
        err_message = cudaGetErrorString(err);
        cout << err_message << endl;
        return NULL;
    }

    /* (B) */
    int blocks_number = get_number_of_needed_blocks(numKeys, maximum_number_of_threads_per_block);

    /* (C) */
    device_hashmap_t *device_hashmap;
    copy_buckets_info_host_device(curr_buckets, max_buckets, NULL, &maximum_buckets);
    compute_hashmap_device(&device_hashmap, maximum_buckets, NULL, hashmap);

    /* (D) */
	kernel_ht_get<<<blocks_number, maximum_number_of_threads_per_block>>>(device_keys, device_values, numKeys, device_hashmap);
    cudaError_t err_sync = cudaDeviceSynchronize();
    if(err_sync != cudaSuccess) {
        const char *err_message = cudaGetErrorString(err_sync);
        cout << err_message;
        return NULL;
    }

    /* (E) */    
    cudaMemcpy(values, device_values, sizeof(int) * numKeys, cudaMemcpyDeviceToHost);

    glbGpuAllocator->_cudaFree(device_keys);
    glbGpuAllocator->_cudaFree(device_values);
    glbGpuAllocator->_cudaFree(maximum_buckets);
    glbGpuAllocator->_cudaFree(device_hashmap);

    return values;
}

