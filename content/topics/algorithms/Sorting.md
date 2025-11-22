---
category: "notes/algorithms"
title: "Sorting"
date: "2024-06-01"
description: "Notes on Sorting Algorithms"
tags: []
---
[cool lc post](https://leetcode.com/problems/sort-an-array/solutions/448903/c-merge-insertion-bubble-quick-heap/)
#### Bubble Sort
O(n) best case, O(n^2) worst case
This is easy to remember, think of bubbles bubbling up some water, every bubble is two adjacent elements swapped.
You start at the start of the array, you swap adjacent elements that are out of order. You keep doing this until all elements are in order.

```cpp
void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    // Loop through the array for bubble sort passes.
    for (int i = 0; i < n - 1; ++i) {
        // Inner loop to compare and swap elements.
        for (int j = 0; j < n - i - 1; ++j) {
            // Compare and swap if elements are in the wrong order.
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}
```

#### Insertion Sort
O(n) best case, O(n\^2) worst case. 
Like the name implies, we take an unsorted number and then insert it into its sorted position.
We do this for each number until the end.
```cpp
vector<int> insertionSort(vector<int>& nums) {
	// check number behind, and if it's greater, move it forward
	for(int i = 1; i < nums.size(); i++){
		int key = nums[i];
		int j = i - 1;
		while(j >= 0 && nums[j] > key){
			nums[j+1] = nums[j];
			j--;
		}
		nums[j+1] = key;
	}
	return nums;
}
```

#### Selection Sort
O(n^2) time complexity
Here, we find the smallest element, put it in the first position. Then we find the second smallest element, put it in the second position, etc. until the end.
```cpp
vector<int> insertionSort(vector<int>& nums) {
	for(int i = 0; i < nums.size(); i++){
		int minIdx = i;
		for(int j = i+1; j < nums.size(); j++){
			if(nums[j] < nums[minIdx]) minIdx = j;
		}
		
		swap(nums[i], nums[minIdx]);
	}
	return nums;
}
```


#### Quick Sort
O(n) for 3-way partition or O(nlogn) normally best case, O(n^2) worst case
It's a form of divide and conquer, you choose a pivot, and partition the remaining elements into two sub-arrays depending on if they're smaller or greater than the pivot.
```cpp
int partitionArray(vector<int> &nums, int low, int high) {
	if(low >= high) return -1;
	int pivot = low, l = pivot + 1, r = high;
	while(l <= r){
		if(nums[l] < nums[pivot]) l++;
		else if(nums[r] >= nums[pivot]) r--;
		else swap(nums[l], nums[r]);
	}
	swap(nums[pivot], nums[r]);
	return r; // new pivot
}

void quickSort(vector<int>& nums, int low, int high){
	if(low >= high) return;
	int pivot = partitionArray(nums, low, high);
	quickSort(nums, low, pivot);
	quickSort(nums, pivot + 1, high);
}    
```

#### Merge Sort
O(nlogn) time complexity, O(n) space.
Merge sort is a divide-and-conquer sorting algorithm. Basically divide the data set into smaller and smaller sub-arrays until it's easy to sort, then merge the sorted subarrays back into a larger sorted array.

```cpp
	void merge(vector<int>& nums, int l, int m, int r){
        vector<int> tmp(r - l + 1);
        int i = l; // index for left subarray
        int j = m + 1; // index for right subarray
        int k = 0; // index for temporary array
        while(i <= m && j <= r){
            if(nums[i] <= nums[j]) tmp[k++] = nums[i++]; 
            else tmp[k++] = nums[j++];
        }
        while(i <= m) tmp[k++] = nums[i++];
        while(j <= r) tmp[k++] = nums[j++]; 
        for(i = 0; i < k; i++) nums[l + i] = tmp[i];
    }
	
	// mergeSort(nums, 0, nums.size() - 1);
    void mergeSort(vector<int>& nums, int l, int r){
        if(l >= r) return;
        int m = l + (r - l) / 2; //middle index, same as (l+r)/2
        mergeSort(nums, l, m);
        mergeSort(nums, m + 1, r);
        merge(nums, l, m, r);
    }
```

#### Heap Sort
O(nlogn) time, O(n) space. on average slower than mergesort but takes a bit less space.
Transform the array into a max-heap. How to do this?
- Start from the first non-leaf node (considering the array is a tree, start in the middle, because the numbers after that will obviously be leaves)
- Run siftDown() on each element. siftDown() checks that the element is bigger than both children. if it's not, sift it down! 

Once you have a maxheap, it's simple, start from the end and swap the end with the biggest element. After that, make sure to sift down again from the root.

```cpp
vector<int> sortArray(vector<int>& nums) {
	for(int i = nums.size() / 2 - 1; i >= 0; i--){
		siftDown(nums, nums.size(), i);
	}

	for(int i = nums.size() - 1; i >= 0; i--){
		swap(nums[0], nums[i]);
		// notice how n is now i. 
		// we dont want to siftDown the already sorted numbers ofc
		siftDown(nums, i, 0); 
	}

	return nums;
}

void siftDown(vector<int>& nums, int n, int i){
	int max = i, left = i * 2 + 1, right = i * 2 + 2;
	if(left < n && nums[left] > nums[max]) max = left;
	if(right < n && nums[right] > nums[max]) max = right;
	
	if(max != i){
		swap(nums[max], nums[i]);
		siftDown(nums, n, max);
	}
}
```