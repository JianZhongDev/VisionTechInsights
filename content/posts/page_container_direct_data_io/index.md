---
author: "Jian Zhong"
title: "PageContainer: Fast, Direct Data I/O Without OS Buffering"
date: "2024-06-13"
description: "A comprehensive guide on how to build a C++ data structure enabling direct data IO without OS buffering"
tags: ["hardware programming", "C++"]
categories: ["hardware programming"]
series: ["hardware programming"]
aliases: ["cpp-pagecontainer_direct_data_io"]
cover:
   image: "images/page_container_direct_data_io/PageContainerDataStructure.png"
   caption: "[cover image] data structure of PageContainer (image credit: Jian Zhong)"
ShowToc: true
TocOpen: false
# math: true
ShowBreadCrumbs: true
---

When creating high-speed data streaming applications, it's important to avoid unnecessary data transfer to keep things fast and efficient. Operating systems (OS) automatically buffer file input/output (I/O) in the computer's memory. However, many data streaming applications already have their own buffering steps, making the OS's additional buffering unnecessary. Disabling this OS buffering allows direct control of data transfer, but it requires the application to access data in sizes that are multiples of the system page size (or disk sector size).

This blog post will show you how to build a C++ data structure called `PageContainer` that lets you access data without the OS buffering. You can find the ready-to-use source code for `PageContainer` in my [Github repository](https://github.com/JianZhongDev/PageContainer).(URL: https://github.com/JianZhongDev/PageContainer ) 


## Container Class Requirements

For applications that need to read and write large amounts of data without OS buffering, the data container should meet the following requirements:
- The buffer size should be a multiple of the page size.
- It should be able to store data whose size does not perfectly match multiples of the page size
- It should be capable of saving multiple containers within a large file.

## The PageContainer Class

To meet the requirements for accessing data without OS buffering, we can create a `PageContainer` class. The following sections will explain each part of this class in detail.


```C++ {linenos=true}
namespace Container{
	typedef int errflag_t;
	static const errflag_t ERR_NULL = 0;
	static const errflag_t SUCCESS = 1;
	static const errflag_t ERR_FAILED = -1;
	static const errflag_t ERR_MEMERR = -2;

	// return errflag for fast error handling
	template<typename dtype>
	class PageContainer {
	private:
		void* buffer_p = nullptr;
		size_t* buffer_size_p = nullptr;
		dtype* data_p = nullptr;
		size_t* data_size_p = nullptr;
		bool allocated_buffer = false;

		// assign data pointers insider buffer
		errflag_t assign_pointers() {
			if (this->buffer_p != nullptr) {
				this->buffer_size_p = (size_t*)this->buffer_p;
				this->data_size_p = ((size_t*)this->buffer_p) + 1;
				this->data_p = (dtype*)(((size_t*)this->buffer_p) + 2);
				return SUCCESS;
			}
			return ERR_FAILED;
		}

		// initialize buffer and data pointers
		errflag_t init_buffer(size_t buffer_size) {
			errflag_t err_flag = ERR_NULL;
			if (this->buffer_p == nullptr) {
				// allocate buffer
				this->buffer_p = malloc(buffer_size);
				if (this->buffer_p == nullptr) return ERR_MEMERR;
				memset(this->buffer_p, 0, buffer_size);
				// assign pointers
				err_flag = this->assign_pointers();
				// update type
				*(this->buffer_size_p) = buffer_size;
				*(this->data_size_p) = 0;
			}
			this->allocated_buffer = true;
			return err_flag;
		}

		// free allocated buffer
		errflag_t free_buffer() {
			if (this->allocated_buffer) { // free buffer only when object allocated the buffer.
				if (this->buffer_p != nullptr) {
					free(this->buffer_p);
				}
			}
			return SUCCESS;
		}

	public:
		// constructor from existing buffer
		PageContainer(void* input_buffer_p, bool new_buffer = true) {
			errflag_t errflag = ERR_NULL;
			size_t buffer_size = *((size_t*)input_buffer_p);
			if (new_buffer) {
				errflag = this->init_buffer(buffer_size);
				memcpy(this->buffer_p, input_buffer_p, buffer_size);
			}
			else {
				this->buffer_p = input_buffer_p;
				errflag = this->assign_pointers();
			}
			// throw error when failed
			if (errflag != SUCCESS) {
				throw std::runtime_error("PageContainer failed to init. Error flag = " + std::to_string(errflag));
			}
		}

		// constructor by giving max data size
		PageContainer(size_t capacity, size_t page_size) {
			errflag_t errflag = ERR_NULL;
			size_t nof_pages = (capacity + 2 * sizeof(size_t)) / page_size;
			if ((capacity + 2 * sizeof(size_t)) % page_size > 0) nof_pages += 1;
			size_t buffer_size = nof_pages * page_size;
			errflag = this->init_buffer(buffer_size);
			// throw error when failed
			if (errflag != SUCCESS) {
				throw std::runtime_error("PageContainer failed to init. Error flag = " + std::to_string(errflag));
			}
		}

		// constructor by giving buffer size
		PageContainer(size_t buffer_size) {
			errflag_t errflag = ERR_NULL;
			errflag = this->init_buffer(buffer_size);
			// throw error when failed
			if (errflag != SUCCESS) {
				throw std::runtime_error("PageContainer failed to init. Error flag = " + std::to_string(errflag));
			}
		}

		// destuctor frees memory
		~PageContainer() {
			this->free_buffer();
		}

		// get the buffer pointer and size
		errflag_t get_buffer(void** buf_pp, size_t* buf_size_p) {
			*buf_pp = this->buffer_p;
			*buf_size_p = *(this->buffer_size_p);
			return SUCCESS;
		}

		// get data pointer 
		errflag_t get_data_p(dtype** data_pp) {
			*data_pp = this->data_p;
			return SUCCESS;
		}

		// get capacity
		errflag_t get_capacity(size_t* capacity_p) {
			size_t capacity = *(this->buffer_size_p) - 2;
			*capacity_p = capacity;
			return SUCCESS;
		}

		// get data size
		errflag_t get_data_size(size_t* data_size_p) {
			*data_size_p = *(this->data_size_p);
			return SUCCESS;
		}

		// set data size
		errflag_t set_data_size(size_t data_size) {
			errflag_t err_flag = ERR_NULL;
			size_t capacity = 0;
			
			err_flag = this->get_capacity(&capacity);
			if (err_flag != SUCCESS) return err_flag;

			if (data_size > capacity) {
				return ERR_FAILED;
			}

			*(this->data_size_p) = data_size;

			return SUCCESS;
		}

	};
}
```

### Data structure

As shown in the **cover image** of this post, the main storage space of the `PageContainer` is a buffer sized as a multiple of the system page size. The first `sizeof(size_t)` bytes (8 bytes on 64-bit systems, 4 bytes on 32-bit systems) store the total size of the buffer. The next `sizeof(size_t)` bytes store the size of the valid data. The remaining space is used for data storage. Pointers are assigned for the entire buffer, the buffer size, the data size, and the data storage area.

This data structure setup allows data access without OS buffering and lets you store data that isn't an exact multiple of the page size. By storing the buffer size and data size for each `PageContainer` object, you can store and access multiple `PageContainer` buffers within a single file without needing extra metadata.

### Creating and deleting a PageContainer

When creating a PageContainer object, it allocates a buffer and organizes it based on the structure described earlier. The methods `assign_pointers()` and `init_buffer()` handle the allocation of the buffer and set up the necessary pointers.


```C++ {linenos=true}
// assign data pointers insider buffer
errflag_t assign_pointers() {
	if (this->buffer_p != nullptr) {
		this->buffer_size_p = (size_t*)this->buffer_p;
		this->data_size_p = ((size_t*)this->buffer_p) + 1;
		this->data_p = (dtype*)(((size_t*)this->buffer_p) + 2);
		return SUCCESS;
	}
	return ERR_FAILED;
}

// initialize buffer and data pointers
errflag_t init_buffer(size_t buffer_size) {
	errflag_t err_flag = ERR_NULL;
	if (this->buffer_p == nullptr) {
		// allocate buffer
		this->buffer_p = malloc(buffer_size);
		if (this->buffer_p == nullptr) return ERR_MEMERR;
		memset(this->buffer_p, 0, buffer_size);
		// assign pointers
		err_flag = this->assign_pointers();
		// update type
		*(this->buffer_size_p) = buffer_size;
		*(this->data_size_p) = 0;
	}
	this->allocated_buffer = true;
	return err_flag;
}
```

With the methods for buffer allocation and arrangement in place, the constructors of the PageContainer can be defined simply as follows:


```C++ {linenos=true}
// constructor by giving buffer size
PageContainer(size_t buffer_size) {
	errflag_t errflag = ERR_NULL;
	errflag = this->init_buffer(buffer_size);
	// throw error when failed
	if (errflag != SUCCESS) {
		throw std::runtime_error("PageContainer failed to init. Error flag = " + std::to_string(errflag));
	}
}

// constructor by giving max data size
PageContainer(size_t capacity, size_t page_size) {
	errflag_t errflag = ERR_NULL;
	size_t nof_pages = (capacity + 2 * sizeof(size_t)) / page_size;
	if ((capacity + 2 * sizeof(size_t)) % page_size > 0) nof_pages += 1;
	size_t buffer_size = nof_pages * page_size;
	errflag = this->init_buffer(buffer_size);
	// throw error when failed
	if (errflag != SUCCESS) {
		throw std::runtime_error("PageContainer failed to init. Error flag = " + std::to_string(errflag));
	}
}
```

Here, the first constructor accepts a directly calculated buffer size (`buffer_size`) from external sources. The second constructor takes expected maximum data size (`capacity`, in bytes) and system page size (`page_size`, in bytes) as inputs, calculating the minimum required buffer size accordingly.

In some applications, such as when loading a `PageContainer` buffer from a file or using intermediate buffering, there may already be memory allocated to store the buffer. In these cases, where pre-existing buffers are available, `PageContainer` also provides constructors that use these buffers directly, without allocating additional space.

```C++ {linenos=true}
// constructor from existing buffer
PageContainer(void* input_buffer_p, bool new_buffer = true) {
	errflag_t errflag = ERR_NULL;
	size_t buffer_size = *((size_t*)input_buffer_p);
	if (new_buffer) {
		errflag = this->init_buffer(buffer_size);
		memcpy(this->buffer_p, input_buffer_p, buffer_size);
	}
	else {
		this->buffer_p = input_buffer_p;
		errflag = this->assign_pointers();
	}
	// throw error when failed
	if (errflag != SUCCESS) {
		throw std::runtime_error("PageContainer failed to init. Error flag = " + std::to_string(errflag));
	}
}
```

To properly manage the dynamically allocated buffer, we have also defined the `free_buffer()` method as follows.

```C++ {linenos=true}
// free allocated buffer
errflag_t free_buffer() {
	if (this->allocated_buffer) { // free buffer only when object allocated the buffer.
		if (this->buffer_p != nullptr) {
			free(this->buffer_p);
		}
	}
	return SUCCESS;
}
```

In the `PageContainer` destructor, the buffer is freed if it was allocated during construction.

```C++ {linenos=true}
// destuctor frees memory
~PageContainer() {
	this->free_buffer();
}
```

### Accessing data and buffer

In data streaming applications, it's common to have APIs that accept pointers to storage spaces for reading and writing data. Here's a typical pesudo code of how such an API might be structured:

```C++ {linenos=true}
errorflag_t write_data(handle_t file_handle, void* src_buffer, size_t bytes_to_write);
errorflag_t read_data(handle_t file_handle, void* dst_buffer, size_t bytes_to_read, size_t* bytes_retrieved);
```
Examples of such APIs include functions like [`WriteFile`](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-writefile) and [`ReadFile`](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-readfile) provided by the Windows API.

```C++ {linenos=true}
BOOL WriteFile(
  [in]                HANDLE       hFile,
  [in]                LPCVOID      lpBuffer,
  [in]                DWORD        nNumberOfBytesToWrite,
  [out, optional]     LPDWORD      lpNumberOfBytesWritten,
  [in, out, optional] LPOVERLAPPED lpOverlapped
);

BOOL ReadFile(
  [in]                HANDLE       hFile,
  [out]               LPVOID       lpBuffer,
  [in]                DWORD        nNumberOfBytesToRead,
  [out, optional]     LPDWORD      lpNumberOfBytesRead,
  [in, out, optional] LPOVERLAPPED lpOverlapped
);
```

`PageContainer` offers the following methods to access data stored within it following such pointer data IO scheme.

```C++ {linenos=true}
// get data pointer 
errflag_t get_data_p(dtype** data_pp) {
	*data_pp = this->data_p;
	return SUCCESS;
}

// get capacity
errflag_t get_capacity(size_t* capacity_p) {
	size_t capacity = *(this->buffer_size_p) - 2;
	*capacity_p = capacity;
	return SUCCESS;
}

// get data size
errflag_t get_data_size(size_t* data_size_p) {
	*data_size_p = *(this->data_size_p);
	return SUCCESS;
}

// set data size
errflag_t set_data_size(size_t data_size) {
	errflag_t err_flag = ERR_NULL;
	size_t capacity = 0;
			
	err_flag = this->get_capacity(&capacity);
	if (err_flag != SUCCESS) return err_flag;

	if (data_size > capacity) {
		return ERR_FAILED;
	}

	*(this->data_size_p) = data_size;

	return SUCCESS;
}
```

The `get_data_p()` method provides a pointer to the storage space where data is stored. The `get_capacity()` method provides the maximum amount of data, in bytes, that the `PageContainer` can hold. The `get_data_size()` method provides the current size of the stored data in bytes. The `set_data_size()` method provides the size of the stored data when new data is written into the `PageContainer`.

For reading and writing operations that require data sizes to be multiples of the page size, `PageContainer` offers the following method to access the entire allocated buffer.

```C++ {linenos=true}
// get the buffer pointer and size
errflag_t get_buffer(void** buf_pp, size_t* buf_size_p) {
	*buf_pp = this->buffer_p;
	*buf_size_p = *(this->buffer_size_p);
	return SUCCESS;
}
```

## Using PageContainer

Here's a quick example demonstrating how to use the `PageContainer`:

```C++ {linenos=true}
void main() {
	std::wstring file_path = L"test_file.bin";
	HANDLE file_handle = INVALID_HANDLE_VALUE;
	size_t page_size = 4 * 1024; // initial 4KB page size
	Container::errflag_t errflag = Container::ERR_NULL;
	DWORD error_flag = NULL;
	DWORD d_error = NULL;

	// get system page size
	SYSTEM_INFO sys_info;
	GetSystemInfo(&sys_info);
	page_size = sys_info.dwPageSize;
	std::cout << "system page size: " << page_size << std::endl;

	// initialize test array
	const size_t array_len = 10;
	float* test_array = new float[array_len];
	size_t data_size = array_len * sizeof(float);

	for (size_t idx = 0; idx < array_len; ++idx) {
		test_array[idx] = rand_float(0.0f, 1.0f, 1000);
	}

	std::cout << "write array: " << carray_to_string(test_array, array_len) << std::endl;

	// create container with page size
	Container::PageContainer<float> write_container(array_len * sizeof(float), page_size);
	
	// copy data into container 
	size_t data_size_to_write = data_size;
	float* write_data_p = nullptr;
	size_t write_capacity = 0;
	errflag = write_container.get_data_p(&write_data_p);
	if (errflag != Container::SUCCESS) {
		std::cout << "ERR:\t failed to access data pointer." << std::endl;
	}
	write_container.get_capacity(&write_capacity);
	memcpy(write_data_p, test_array, data_size_to_write);
	write_container.set_data_size(data_size_to_write);

	// write data to file (bypassing system buffer)
	// open file
	file_handle = CreateFileW(
		file_path.c_str(),
		GENERIC_WRITE | GENERIC_READ,
		0,
		NULL,
		CREATE_ALWAYS,
		FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, 
		NULL);
	if (file_handle == INVALID_HANDLE_VALUE) {
		std::cout << "ERR:\t failed to create file." << std::endl;
		return;
	}

	void* write_buffer_p = nullptr;
	size_t write_buffer_size = 0;
	write_container.get_buffer(&write_buffer_p, &write_buffer_size);

	// write data
	error_flag = WriteFile(file_handle, write_buffer_p, write_buffer_size, NULL, NULL);
	d_error = GetLastError();
	if (error_flag == FALSE && d_error != ERROR_IO_PENDING) {
		std::cout << "ERR:\t error in write file, error code = " << d_error << std::endl;
		return;
	}

	// close file
	CloseHandle(file_handle);
	file_handle = INVALID_HANDLE_VALUE;

	std::cout << "data write to: ";
	std::wcout << file_path;
	std::cout << std::endl;

	// read data from file
	void* read_buffer_p = nullptr;
	size_t read_buffer_size = 0;
	DWORD bytes_read = 0;

	// open file
	file_handle = CreateFileW(
		file_path.c_str(),
		GENERIC_READ,
		0,
		NULL,
		OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL,
		NULL);
	if (file_handle == INVALID_HANDLE_VALUE) {
		std::cout << "ERR:\t failed to create file." << std::endl;
		return;
	}
	// get buffer size
	error_flag = ReadFile(file_handle, &read_buffer_size, sizeof(size_t), &bytes_read, NULL);
	d_error = GetLastError();
	if (error_flag == FALSE && d_error != ERROR_IO_PENDING) {
		std::cout << "ERR:\t error in read file, error code = " << d_error << std::endl;
		return;
	}

	std::cout << "read_buffer_size = " << read_buffer_size << std::endl;
	Container::PageContainer<float> read_container(read_buffer_size);
	read_container.get_buffer(&read_buffer_p, &read_buffer_size);

	// set file pointer to start of the buffer 
	SetFilePointer(file_handle, 0, NULL, FILE_BEGIN);
	d_error = GetLastError();
	if (d_error != 0) {
		std::cout << "ERR:\t error set file pointer, error code = " << d_error << std::endl;
		return;
	}

	// load buffer from the file
	error_flag = ReadFile(file_handle, read_buffer_p, read_buffer_size, &bytes_read, NULL);
	d_error = GetLastError();
	if (error_flag == FALSE && d_error != ERROR_IO_PENDING) {
		std::cout << "ERR:\t error in read file, error code = " << d_error << std::endl;
		return;
	}

	// close file
	CloseHandle(file_handle);
	file_handle = INVALID_HANDLE_VALUE;

	std::cout << "data read from: ";
	std::wcout << file_path;
	std::cout << std::endl;

	// display read data
	float* read_arr = nullptr;
	size_t read_data_size = 0;
	size_t read_arr_len = 0;
	read_container.get_data_p(&read_arr);
	read_container.get_data_size(&read_data_size);
	read_arr_len = read_data_size / sizeof(float);
	
	std::cout << "read array: " << carray_to_string(read_arr, read_arr_len) << std::endl;

}
```

In this example, we start by creating a `PageContainer` object (`write_container`) to store a float test array. The size of this array doesn't match the system's page size. We use the `memcpy()` function explicitly in the example to copy data into the `PageContainer` object.
Next, we create a file with the `FILE_FLAG_NO_BUFFERING` flag, which disables Windows' OS file buffering. We then write the entire buffer of the `PageContainer` object to this file and close it.

Lastly, we reopen the saved file to read the buffer size. Using this size, we create a new `PageContainer` object (`read_container`) and load the buffer from the file. Finally, we retrieve the array from the `read_container`.

The output displayed in the terminal for the demo code above is as follows:

```{linenos=true}
system page size: 4096
write array: {0.041, 0.467, 0.334, 0.5, 0.169, 0.724, 0.478, 0.358, 0.962, 0.464}
data write to: test_file.bin
read_buffer_size = 4096
data read from: test_file.bin
read array: {0.041, 0.467, 0.334, 0.5, 0.169, 0.724, 0.478, 0.358, 0.962, 0.464}
```

## Conclusion

By allocating a memory space sized in multiples of page sizes and organizing it into sections for storing buffer size, data size, and actual data, we created the `PageContainer` class. This setup enables direct reading and writing of data without relying on OS buffering.

## Citation

If you found this article helpful, please cite it as:
> Zhong, Jian (June 2024). PageContainer: Fast, Direct Data I/O Without OS Buffering. Vision Tech Insights. https://jianzhongdev.github.io/VisionTechInsights/posts/page_container_direct_data_io/.

Or

```html
@article{zhong2024pagecontainer,
  title   = "PageContainer: Fast, Direct Data I/O Without OS Buffering",
  author  = "Zhong, Jian",
  journal = "jianzhongdev.github.io",
  year    = "2024",
  month   = "June",
  url     = "https://jianzhongdev.github.io/VisionTechInsights/posts/building_a_configuration_file_parser_with_cpp/."
}
```




