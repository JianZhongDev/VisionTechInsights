---
author: "Jian Zhong"
title: "Building a Configuration File Parser with C++"
date: "2024-04-21"
description: "A comprehensive guide on how to build a configuration file parser with C++ from scratch."
tags: ["hardware programming", "C++"]
categories: ["hardware programming"]
series: ["hardware programming"]
aliases: ["cpp-config-file-parser"]
cover:
   image: images/building_a_configuration_file_parser_with_cpp/CppConfigFileModuleStructure.png
   caption: "[cover image] diagram of the configuration file parser (image credit: Jian Zhong)"
ShowToc: true
TocOpen: false
# math: true
ShowBreadCrumbs: true
---

Configuration files are commonly used to adjust settings in computer programs. I'm presently developing a configuration file parser for my high-speed data acquisition system using C++. Along the way, I've discovered some useful techniques involving C++ generics and inheritance that streamline coding. Therefore, I decided to document these tricks in the hope that they'll be beneficial to others. You can find the ready-to-use source code for this configuration file parser in my [GitHub repository](https://github.com/JianZhongDev/CppConfigFile). (URL: https://github.com/JianZhongDev/CppConfigFile.)

## Configuration Files

According to [Wikipedia](https://en.wikipedia.org/wiki/Configuration_file), configuration files are files used to set up the parameters and initial settings for computer programs. A configuration file parser is a piece of program that allows saving program settings to and loading them from configuration files. Configuration files are very handy when users need to provide certain configuration settings to the program before starting it. In my research, I've also discovered the convenience of having a configuration file parser to record the configuration settings of my experiments. This enables me to quickly switch between different software settings for various applications.

## Requirement Analysis

For a data acquisition program, users often need to fine-tune settings to optimize performance for their specific needs. This includes things like timing delays, filtering coefficients, and switching between different data processing methods. These settings are stored as variables with various data types (like numbers, strings, and arrays) in the software. So, the configuration file parser has to handle a wide range of data types.

We also want the configuration file to be easily readable and editable by users, so it needs to be in a human-readable text format. This means the parser should be able to convert variables to strings and back again.

Plus, it'd be great if users could add comments to the configuration file to keep track of changes.

In summary, here's what the configuration file parser needs to do:

1. Store multiple setting variables and their values.
2. Handle values with different data types.
3. Convert variables to strings, and update values from strings.
4. Save settings to a text file that's easy for humans to read.
5. Update variable values from the text file.
6. Process configuration files with comments.

## Data Structure and Algorithm Design

Once we've nailed down the requirements, we can begin designing the data structures to meet them.

### Generic Entry

To handle the task of storing variables with different data types (requirement 2), we can develop our own custom class called `GenericEntry`. This class will enable us to access the data within it using the `set()` and `get()` methods for writing and reading data, respectively. Since different data types require different methods for reading and writing, we make these `set()` and `get()` methods virtual and require subclasses to implement them. The `GenericEntry` class also includes a  `type_name` member and a `get_typename()` method to record the data type and verify data types.

For converting data to and from strings (requirement 3), considering that various data types require different approaches for this conversion, the `GenericEntry` class provides `write_val_string()` and `read_val_string()` methods. These methods facilitate converting the data value to a string and vice versa.

```C++ {linenos=true}
// Base type of generic entry
class GenericEntry {
protected:
	std::string type_name;
public:
	//TODO: could potentially change the return type from void to int and return error flag
	//TODO: could potentially use type_index instead of hardcoded string as the type identifier

	GenericEntry() {
		// set up the type_name in the constructor
		this->type_name = "generic_entry";
	}

	// Set value of the entry
	virtual void set() {
		//Override this method in subclass 
	}

	// Get value of the entry
	virtual void get() {
		//Override this method in subclass 
	}

	// Return string of the type name
	void get_typename(std::string* dst_string) {
		*dst_string = this->type_name;
	}

	// Write the value of the entry into string
	virtual void write_val_string(std::string* dst_string) {
		//Override this method in subclass 
	}

	// Read value of the entry from string
	virtual void read_val_string(const std::string& src_string) {
		//Override this method in subclass 
	}
};

```

Once we've set up the most basic entry type, we create a more specialized subclass named `TypedEntry`. Leveraging the generics template feature, we implement the `set()` and `get()` functions. However, because custom classes, iterable types, and primitive data types (like `int` and `unsigned int`) require unique approaches for converting their data to strings, we leave the `write_val_string()`and `read_val_string()` methods for future implementation in more specialized subclasses.

```C++ {linenos=true}
// Entry of generic type definition
template<typename data_t>
class TypedEntry : public GenericEntry {
protected:
	data_t data;
public:
	//Constructor without initial value
	TypedEntry() {
		this->type_name = "typed_entry";
	}
	//Constructor with initial value
	TypedEntry(data_t data) {
		this->TypedEntry();
		this->data = data_t(data);
	}

	// Implemented set entry value method
	template<typename data_t>
	void set(const data_t& data) {
		this->data = data_t(data);
	}

	// Implemented get entry value method
	template<typename data_t>
	void get(data_t* data_p) {
		*data_p = data_t(this->data);
	}

	virtual void write_val_string(std::string* dst_string) {
		//Override this method in subclass 
	}

	virtual void read_val_string(const std::string& src_string) {
		//Override this method in subclass
	}

};

```

Primitive types (like `int` and `unsigned int`) have straightforward methods for converting between data and strings. We can implement their entry classes like this: For each specific primitive type, we simply inherit from the primitive type entry and specify the type name in the constructors.

```C++ {linenos=true}
// Entries with primitive type  
template<typename data_t>
class PrimitiveTypeEntry : public TypedEntry<data_t> {
	// NOTE: Only need to define contructor giving type_name in the subclasses
public:
	PrimitiveTypeEntry() {
		this->type_name = "primitivetype_entry";
	}
	PrimitiveTypeEntry(const data_t& data) {
		this->PrimitiveTypeEntry();
		this->data = data_t(data);
	}
	virtual void write_val_string(std::string* dst_string) {
		*dst_string = std::to_string(this->data);
	}
	virtual void read_val_string(const std::string& src_string) {
		if (std::is_fundamental<data_t>::value) { // validate the data type is primitive data type
			std::stringstream(src_string) >> this->data; // use stringstream to convert value string to value
		}
	}
};

// Entries with int type
class IntEntry : public PrimitiveTypeEntry<int> {
public:
	IntEntry(int data = 0) {
		this->type_name = "int";
		this->data = data;
	}
};

```

We can apply a similar approach to define types for vectors as well.


```C++ {linenos=true}
// Vector class with primitive data type
template<typename data_t>
class VectorPrimitiveTypeEntry : public TypedEntry<std::vector<data_t>> {
	// NOTE: Only need to define contructor giving type_name in the subclasses
protected:
	//NOTE: data string format: {val0, val1, val2}
	std::string str_dl = ",";
	std::string str_enclosure[2] = { "{", "}" };
public:
	VectorPrimitiveTypeEntry() {
		this->type_name = "vector_primitivetype";
	}

	VectorPrimitiveTypeEntry(const std::vector<data_t>& data) {
		this->VectorPrimitiveTypeEntry();
		this->data = std::vector<data_t>(data);
	}

	virtual void write_val_string(std::string* dst_string) {
		std::stringstream result_strstream;
		unsigned data_len = this->data.size();
		unsigned count = 0;

		// iterate through data vector
		result_strstream << this->str_enclosure[0];
		for (auto itr = this->data.begin(); itr != this->data.end(); ++itr) {
			result_strstream << std::to_string(*itr);
			count++;
			if (count < data_len) result_strstream << this->str_dl;
		}
		result_strstream << this->str_enclosure[1];

		*dst_string = result_strstream.str();
	}

	virtual void read_val_string(const std::string& src_string) {
		// remove '{', '}', and '_'
		std::string tmp_str = helper_extract_string_between_enclosure(src_string, str_enclosure[0], str_enclosure[1]);
		tmp_str = helper_clean_tailheadchars_string(tmp_str, std::unordered_set<char>{' '});
		// extract value string for each element
		std::vector<std::string> val_strs = helper_split_string_with_delimiter(tmp_str, this->str_dl);
		
		if (std::is_fundamental<data_t>::value) { // validate data type
			// iterate through value strings for each element
			this->data.clear();
			for (auto itr = val_strs.begin(); itr != val_strs.end(); ++itr) {
				data_t tmp_val;
				std::stringstream(*itr) >> tmp_val;
				this->data.push_back(tmp_val);
			}
		}
	}
};

// Entry with float vector
class VectorFloatEntry : public VectorPrimitiveTypeEntry<float> {
public:
	VectorFloatEntry(const std::vector<float>& data = { 0.0 }) {
		this->type_name = "vector_float";
		this->data = std::vector<float>(data);
	}
};

```

Since a string is a more specialized class-based data type, we need to define its entry separately.

```C++ {linenos=true}
// Entries with string type
class StringEntry : public TypedEntry<std::string> {
protected:
	// NOTE: value string format: "value_string"
	std::string str_enclosure[2] = { "\"", "\"" };
public:
	StringEntry(const std::string& data = "") {
		this->type_name = "string";
		this->data = data;
	}

	virtual void write_val_string(std::string* dst_string) {
		// Add " " to string
		*dst_string = str_enclosure[0] + std::string(this->data) + str_enclosure[1];
	}

	virtual void read_val_string(const std::string& src_string) {
		// Extract string between " "
		std::string tmp_str = helper_extract_string_between_enclosure(src_string, str_enclosure[0], str_enclosure[1]);
		this->data = std::string(tmp_str);
	}
};
```

### Generic Hashmap

Once we've got our generic entry class ready, we can tackle the task of storing data for multiple settings variables (requirement 1). We can achieve this by using a hash map (`std::unordered_map`), where we map the name of each setting variable to the entry storing its value. One important thing to remember is that when defining the hashmap, the value should be declared as a pointer to the base class. This prevents any issues where a subclass might get casted into the base class when adding it to the hashmap.

```C++ {linenos=true}
typedef std::unordered_map<std::string, GenericEntry*> GenHashMap;
GenHashMap test_genhashmap;
```

With this generic hashmap setup, adding setting variables and entries is straightforward:

```C++ {linenos=true}
// initialize generic hash map
test_genhashmap["int_val"] = new IntEntry(1);
```

Converting the entry to and from a string is as simple as this:

```C++ {linenos=true}
// update entry with string
test_map["int_val"]->read_val_string("-1");
// convert entry value into string
std::string tmp_valstr;
test_map["int_val"]->write_val_string(&tmp_valstr);
```

After casting the entry to its subclass, we can easily set and retrieve values within the entry.

```C++ {linenos=true}
// set value of entry
((IntEntry*)test_genhashmap["int_val"])->set(-1);
// get value from entry
int tmp_int;
((IntEntry*)test_genhashmap["int_val"])->get(&tmp_int);
```

Furthermore, we can easily determine the type of the entry by calling the get_typename() method.

```C++ {linenos=true}
// get type name string from entry
std::string tmp_typename;
test_map["int_val"]->get_typename(&tmp_typename);
```

To simplify clearing the entire hashmap, I've created the clear_genhashmap() function, outlined below:

```C++ {linenos=true}
typedef int errflag_t;

// delete all the elements in a generic hash map
errflag_t clear_genhashmap(
	GenHashMap& gen_hashmap
) {
	// iterate through the hash map to release all the entries
	for (auto key_val_pair : gen_hashmap) {
		delete key_val_pair.second;
	}
	gen_hashmap.clear();

	return 1;
}
```

Note: If maintaining the order of setting variables in the configuration file is crucial for your application, you can easily achieve this by switching the data type from `std::unordered_map` (hashmap) to `std::ordered_map` (tree-based map). Everything else in the code remains unchanged and can be used as is.

### Saving Configuration Files

Since we've already implemented the string conversion function in the entries, saving the setting parameters to human-readable text files is straightforward. We simply need to iterate through the generic hashmap, saving the name (key), type, and value of each entry. Then, we add entry separators at the end of each entry and dump them into a text file. Additionally, I've included a header string to provide some helpful information in the configuration file.

```C++ {linenos=true}
// pack type name value string into one string
std::string helper_pack_type_name_val_string(
	const std::string& type_string,
	const std::string& name_string,
	const std::string& val_string,
	const std::string& type_name_dl = " ",
	const std::string& name_val_dl = "="
) {
	return type_string + type_name_dl + name_string + name_val_dl + val_string;
}

typedef int errflag_t;

// save generic hash map entries to configuration text file
errflag_t save_genhashmap_to_txt(
	const GenHashMap& gen_hashmap,
	const std::string& dst_file_path,
	std::ios_base::openmode dst_file_openmode = std::ios_base::out,
	const std::string& type_name_dl = " ",
	const std::string& name_val_dl = "=",
	const std::string& entry_stop_str = ";",
	const std::vector<std::string>& default_message_enclousre = {"/*", "*/"},
	const std::string& head_message = ""
) {
	errflag_t err_flag = 0;
	std::ofstream dst_file(dst_file_path, dst_file_openmode);
	if (dst_file.is_open()) {
		// save head message if given
		if (head_message.size() > 0) {
			dst_file << default_message_enclousre[0] + head_message + default_message_enclousre[1] + "\n";
		}
		// iterate though hash map and save all entries
		for (const auto& key_val_pair : gen_hashmap) {
			std::string cur_name_str = key_val_pair.first;
			std::string cur_type_str;
			std::string cur_val_str;
			key_val_pair.second->get_typename(&cur_type_str);
			key_val_pair.second->write_val_string(&cur_val_str);
			// convert type name value to entry string
			std::string cur_entry_str = helper_pack_type_name_val_string(
				cur_type_str,
				cur_name_str,
				cur_val_str,
				type_name_dl,
				name_val_dl
			);
			dst_file << cur_entry_str + entry_stop_str + "\n";
		}
		dst_file.close(); // close file
		err_flag = 1;
	}
	else {
		//std::cout << "ERR:\t Unable to open file. File path = " + dst_file_path << std::endl;
		std::string err_msg = "ERR:\t Unable to open file. File path = " + dst_file_path + "\n";
		std::cout << err_msg;
		err_flag = -1;
	}
	return err_flag;
}
```

### Loading Configuration Files

Reading the setting information from the configuration file involves a bit more complexity. We need to handle comments in the file and avoid mistakenly reading separators within strings of entries with string type. These requirements are addressed by iterating through the entire configuration file string using two pointers.
Here's how it works:
- The faster pointer moves ahead to mark the end of each candidate string while continuously checking the substring.
- The slower pointer sets the start position of each candidate string.
- Depending on the substring, the algorithm behaves as follows:
  - If the substring matches the start separator of a string candidate to ignore, the faster pointer moves forward while ignoring all substrings until it finds the end separator of the ignore string candidate.
  - If the substring matches the start separator of a comment string, the faster pointer continues moving forward while ignoring until it finds the end separator of the comment string candidate. The slower pointer ends up positioned after the end of the comment candidate string so that it's not read in.
  - If the substring matches the end separator of an entry string candidate, the substring between the slow and fast pointers is saved into the result vector. This indicates that we've found the string for the setting parameter entry. The entry string candidate undergoes some cleaning processes to remove any extra spaces and newline characters at both ends.

```C++ {linenos=true}
// extract entry strings from complicated strings
std::vector<std::string> helper_extract_entrystr(
	const std::string& src_string,
	const std::string& entry_stop_str = ";", //string indicates the end of an entry string
	std::unordered_map<std::string, std::string> ignore_left_to_right_map = { {"//", "\n"}, {"/*", "*/"} }, //string parts between "left" and "right" to ignore
	std::unordered_map<std::string, std::string> include_left_to_right_map = { {"\"", "\""} } //string parts between "left" and "right" to include
) {
	std::size_t slow_idx = 0;
	std::size_t fast_idx = 0;
	std::size_t srcstr_len = src_string.size();
	std::size_t entry_stop_str_len = entry_stop_str.size();

	// count string lengths in the left_to_right map keys
	std::unordered_set<std::size_t> ignore_left_lens;
	for (const auto& itr : ignore_left_to_right_map) {
		ignore_left_lens.insert(itr.first.size());
	}

	// count string lengths in the left_to_right map keys
	std::unordered_set<std::size_t> include_left_lens;
	for (const auto& itr : include_left_to_right_map) {
		include_left_lens.insert(itr.first.size());
	}

	// itrate through src_string to find all entry strings
	std::vector<std::string> entry_strs;
	while (fast_idx < srcstr_len) {
		// check string between "left" and "right" to include
		for (auto cur_left_len : include_left_lens) {
			if (fast_idx + cur_left_len > srcstr_len) continue;
			std::string cur_left_str = src_string.substr(fast_idx, cur_left_len);
			if (include_left_to_right_map.find(cur_left_str) != include_left_to_right_map.end()) {
				std::string cur_right_str = include_left_to_right_map[cur_left_str];
				std::size_t cur_right_len = cur_right_str.size();
				for (fast_idx += cur_left_len; fast_idx + cur_right_len < srcstr_len; ++fast_idx) {
					if (src_string.substr(fast_idx, cur_right_len) == cur_right_str) break;
				}
				fast_idx += cur_right_len;
			}
		}
		// check string between "left" and "right" to exclude
		for (auto cur_left_len : ignore_left_lens) {
			if (fast_idx + cur_left_len > srcstr_len) continue;
			std::string cur_left_str = src_string.substr(fast_idx, cur_left_len);
			if (ignore_left_to_right_map.find(cur_left_str) != ignore_left_to_right_map.end()) {
				std::string cur_right_str = ignore_left_to_right_map[cur_left_str];
				std::size_t cur_right_len = cur_right_str.size();
				for (fast_idx += cur_left_len; fast_idx + cur_right_len < srcstr_len; ++fast_idx) {
					if (src_string.substr(fast_idx, cur_right_len) == cur_right_str) break;
				}
				fast_idx += cur_right_len;
				slow_idx = fast_idx;
			}
		}
		if (fast_idx + entry_stop_str_len > srcstr_len) break; // reach the end of src_string
		// found complete entry string
		if (src_string.substr(fast_idx, entry_stop_str_len) == entry_stop_str) {
			entry_strs.push_back(src_string.substr(slow_idx, fast_idx - slow_idx));
			slow_idx = fast_idx + 1;
		}
		++fast_idx;
	}

	return entry_strs;
}

```

Once we have the entry strings, we'll break them down into type, name, and value fields. Each string for these fields will be cleaned up, removing any extra spaces and newline characters at both ends. Then, we'll check if the hashmap has a key with the specified name, using it to identify the entries. We'll then examine the value of the entry and update it with the corresponding value string if the type matches.

```C++ {linenos=true}
typedef int errflag_t;

// update generic hash map entries according to configuration text file
errflag_t update_genhashmap_from_txt(
	GenHashMap& gen_hashmap,
	const std::string& src_file_path,
	std::ios_base::openmode src_file_openmode = std::ios_base::in,
	const std::string& type_name_dl = " ",
	const std::string& name_val_dl = "=",
	const std::string& entry_stop_str = ";",
	const std::unordered_map<std::string, std::string>& ignore_left_to_right_map = {{"//", "\n"}, {"/*", "*/"}},
	const std::unordered_map<std::string, std::string>& include_left_to_right_map = {{"\"", "\""}},
	const std::unordered_set<char>& rm_chars = {' ', '\n', '\t'}
) {
	errflag_t err_flag = 0;
	std::ifstream src_file(src_file_path, src_file_openmode);
	if (src_file.is_open()) {
		std::string src_string(
			(std::istreambuf_iterator<char>(src_file)),
			std::istreambuf_iterator<char>()
			);
		std::vector<std::string> entry_strings = helper_extract_entrystr(
			src_string,
			entry_stop_str,
			ignore_left_to_right_map,
			include_left_to_right_map
		);
		for (auto& cur_entry_str : entry_strings) {
			std::string tmp_str;
			std::size_t tmp_str_len = 0;
			//clean up entry string
			tmp_str = helper_bothside_clean_chars(
				cur_entry_str,
				rm_chars
			);
			// split entry string
			std::string type_string;
			std::string name_string;
			std::string value_string;
			helper_split_entrystr_into_type_name_val(
				tmp_str,
				type_name_dl,
				name_val_dl,
				&type_string,
				&name_string,
				&value_string
			);
			// clean up name string
			name_string = helper_bothside_clean_chars(
				name_string,
				rm_chars
			);
			// update entry if name exists
			if (gen_hashmap.find(name_string) != gen_hashmap.end()) {
				type_string = helper_bothside_clean_chars(
					type_string,
					rm_chars
				);
				// update entry if type match
				std::string hp_typename;
				gen_hashmap[name_string]->get_typename(&hp_typename);
				if (type_string == hp_typename) {
					value_string = helper_bothside_clean_chars(
						value_string,
						rm_chars
					);
					gen_hashmap[name_string]->read_val_string(value_string);
				}
				else {
					//std::cout << "ERR:\tType mismatch! " + type_string + " <--> " + hp_typename + "\n";
					std::string err_string = "ERR:\tType mismatch! " + type_string + " <--> " + hp_typename + "\n";
					std::cout << err_string;
				}
			}
			else {
				//std::cout << "ERR:\tName not found! " + name_string + "\n";
				std::string err_string = "ERR:\tName not found! " + name_string + "\n";
				std::cout << err_string;
			}
		}
		err_flag = 1;
	}
	else {
		//std::cout << "ERR:\t Unable to open file. File path = " + src_file_path << std::endl;
		std::string err_string = "ERR:\t Unable to open file. File path = " + src_file_path + "\n";
		std::cout << err_string;
		err_flag = -1;
	}

	return err_flag;
}
```

Since we typically use the same separators and comment notations when writing and reading the configuration file, it makes sense to create a class to store these separators and comment notations for the functions responsible for saving and loading the configuration file.


```C++ {linenos=true}
typedef int errflag_t;

// class for text IO of generic hash map
class GenHashMapIOTxt {
public:
	std::string type_name_dl = " ";
	std::string name_val_dl = "=";
	std::string entry_stop_str = ";";
	std::vector<std::string> default_message_enclousre = { "/*", "*/" };
	std::unordered_map<std::string, std::string> ignore_left_to_right_map = { {"//", "\n"}, {"/*", "*/"} };
	std::unordered_map<std::string, std::string> include_left_to_right_map = { {"\"", "\""} };
	std::unordered_set<char> rm_chars = { ' ', '\n', '\t' };

	// save generic hash map to file
	errflag_t save_to_file(
		const GenHashMap& gen_hashmap,
		const std::string& dst_file_path,
		std::ios_base::openmode dst_file_openmode = std::ios_base::out,
		const std::string& head_message = ""
	) {
		return save_genhashmap_to_txt(
			gen_hashmap,
			dst_file_path,
			dst_file_openmode,
			this->type_name_dl,
			//this->name_val_dl,
			" " + this->name_val_dl + " ",
			this->entry_stop_str,
			this->default_message_enclousre,
			head_message
		);
	}

	// load generic hash map 
	errflag_t update_from_file(
		GenHashMap& gen_hashmap,
		const std::string& src_file_path,
		std::ios_base::openmode src_file_openmode = std::ios_base::in
	) {
		return update_genhashmap_from_txt(
			gen_hashmap,
			src_file_path,
			src_file_openmode,
			this->type_name_dl,
			this->name_val_dl,
			this->entry_stop_str,
			this->ignore_left_to_right_map,
			this->include_left_to_right_map,
			this->rm_chars
		);
	}
};
```

## Unit Test

Finally, we can quickly test the configuration file parser. In the following code, I've used an integer entry, a vector of floats entry, and a string entry as examples.

First, we create the generic hashmap and add initial entries to it. Then, we save the generic hashmap to a text file. Next, we modify the values in the generic map. Finally, we load the values from the configuration file.

```C++ {linenos=true}
void main() {
	GenHashMap test_genhashmap;
	GenHashMapIOTxt test_io_txt;

	// initialize generic hash map
	test_genhashmap["int_val"] = new IntEntry(1);
	test_genhashmap["string_val"] = new StringEntry("Welcome to Vision Tech Insights!");
	test_genhashmap["vector_float_val"] = new VectorFloatEntry({ 0.1, 0.2, 0.3, 0.4, 0.5 });

	std::cout << "====== Initialize gen hashmap ======" << std::endl << std::endl;

	// print all the values in the generic hash map
	std::cout << "--- genhashmap values START ---" << std::endl;
	for (const auto& key_val_pair : test_genhashmap) {
		std::string cur_name_str = key_val_pair.first;
		std::string cur_type_str;
		std::string cur_val_str;
		key_val_pair.second->get_typename(&cur_type_str);
		key_val_pair.second->write_val_string(&cur_val_str);
		std::cout << cur_type_str << " " << cur_name_str << " = " << cur_val_str << std::endl;
	}
	std::cout << "--- genhashmap values END ---" << std::endl;
	std::cout << std::endl;


	// save generic hash map into a text file
	std::cout << "====== Save gen hashmap to text file ======" << std::endl << std::endl;

	// save to txt configuration file
	test_io_txt.save_to_file(
		test_genhashmap,
		"test_config_file.txt",
		std::ios_base::out,
		"This is the test configuration file for Vision Tech Insights blog."
	);


	// save generic hash map into a text file
	std::cout << "====== Set values in gen hashmap ======" << std::endl << std::endl;
	// try to change some of the values
	((IntEntry*)test_genhashmap["int_val"])->set(-1);
	((StringEntry*)test_genhashmap["string_val"])->set(std::string("Hello world!"));
	((VectorFloatEntry*)test_genhashmap["vector_float_val"])->set(std::vector<float>{-0.1, -0.2, -0.3});

	std::cout << "--- genhashmap values START ---" << std::endl;
	for (const auto& key_val_pair : test_genhashmap) {
		std::string cur_name_str = key_val_pair.first;
		std::string cur_type_str;
		std::string cur_val_str;
		key_val_pair.second->get_typename(&cur_type_str);
		key_val_pair.second->write_val_string(&cur_val_str);
		std::cout << cur_type_str << " " << cur_name_str << " = " << cur_val_str << std::endl;
	}
	std::cout << "--- genhashmap values END ---" << std::endl;
	std::cout << std::endl;

	// try to get value from genhashmap
	std::cout << "====== Get values from gen hashmap ======" << std::endl << std::endl;
	int tmp_int;
	((IntEntry*)test_genhashmap["int_val"])->get(&tmp_int);
	std::string tmp_string;
	((StringEntry*)test_genhashmap["string_val"])->get(&tmp_string);
	std::vector<float> tmp_float_vec;
	((VectorFloatEntry*)test_genhashmap["vector_float_val"])->get(&tmp_float_vec);

	std::cout << "--- get values START ---" << std::endl;
	std::cout << "tmp_int = " << tmp_int << std::endl;
	std::cout << "tmp_float_vec = {";
	for (int i_val = 0; i_val < tmp_float_vec.size(); i_val++) {
		std::cout << tmp_float_vec[i_val];
		if (i_val < tmp_float_vec.size() - 1) {
			std::cout << ", ";
		}
	}
	std::cout << "}" << std::endl;
	std::cout << "tmp_string = " << tmp_string << std::endl;
	std::cout << "--- get values END ---" << std::endl;
	std::cout << std::endl;


	// load generic hash map from the text file
	std::cout << "====== Load gen hashmap from text file ======" << std::endl << std::endl;

	// update values according to txt configuration file
	test_io_txt.update_from_file(
		test_genhashmap,
		"test_config_file.txt",
		std::ios_base::in
	);

	std::cout << "--- genhashmap values START ---" << std::endl;
	for (const auto& key_val_pair : test_genhashmap) {
		std::string cur_name_str = key_val_pair.first;
		std::string cur_type_str;
		std::string cur_val_str;
		key_val_pair.second->get_typename(&cur_type_str);
		key_val_pair.second->write_val_string(&cur_val_str);
		std::cout << cur_type_str << " " << cur_name_str << " = " << cur_val_str << std::endl;
	}
	std::cout << "--- genhashmap values END ---" << std::endl;
	std::cout << std::endl;

	// release generic entries and clear gen hashmap
	clear_genhashmap(test_genhashmap);

}
```

The resulting configuration file ("test_config_file.txt") looks like this:

```
/*This is the test configuration file for Vision Tech Insights blog.*/
int int_val = 1;
string string_val = "Welcome to Vision Tech Insights!";
vector_float vector_float_val = {0.100000,0.200000,0.300000,0.400000,0.500000};
```

The output of the execution is as follows:

```
====== Initialize gen hashmap ======

--- genhashmap values START ---
int int_val = 1
string string_val = "Welcome to Vision Tech Insights!"
vector_float vector_float_val = {0.100000,0.200000,0.300000,0.400000,0.500000}
--- genhashmap values END ---

====== Save gen hashmap to text file ======

====== Set values in gen hashmap ======

--- genhashmap values START ---
int int_val = -1
string string_val = "Hello world!"
vector_float vector_float_val = {-0.100000,-0.200000,-0.300000}
--- genhashmap values END ---

====== Get values from gen hashmap ======

--- get values START ---
tmp_int = -1
tmp_float_vec = {-0.1, -0.2, -0.3}
tmp_string = Hello world!
--- get values END ---

====== Load gen hashmap from text file ======

--- genhashmap values START ---
int int_val = 1
string string_val = "Welcome to Vision Tech Insights!"
vector_float vector_float_val = {0.100000,0.200000,0.300000,0.400000,0.500000}
--- genhashmap values END ---

```

## Conclusion

This post offers an example implementation of a configuration file parser using C++. We've developed a generic entry class to handle various data types and convert them into strings. Leveraging inheritance and generics in C++ allows us to reuse code blocks for different data types. Storing entries and their corresponding variable names in a hashmap enables us to manage multiple variables simultaneously. With a classic two-pointer-based string processing algorithm, we can save the hashmap to or load it from the configuration file.

## Citation

If you found this article helpful, please cite it as:
> Zhong, Jian (Apr 2024). Building a Configuration File Parser with C++. Vision Tech Insights. https://jianzhongdev.github.io/VisionTechInsights/posts/building_a_configuration_file_parser_with_cpp/.

Or

```html
@article{zhong2024configfileparsercpp,
  title   = "Building a Configuration File Parser with C++",
  author  = "Zhong, Jian",
  journal = "jianzhongdev.github.io",
  year    = "2024",
  month   = "Apr",
  url     = "https://jianzhongdev.github.io/VisionTechInsights/posts/building_a_configuration_file_parser_with_cpp/."
}
```

## References

[1] "Configuration file." Wikipedia, The Free Encyclopedia. Wikimedia Foundation, Inc. Retrieved April 21, 2024, from https://en.wikipedia.org/wiki/Configuration_file. 



