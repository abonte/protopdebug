#!/usr/bin/env bash

classes=(
  113.Baird_Sparrow
  200.Common_Yellowthroat
  076.Dark_eyed_Junco
  086.Pacific_Loon
  040.Olive_sided_Flycatcher
  037.Acadian_Flycatcher
  026.Bronzed_Cowbird
  001.Black_footed_Albatross
  128.Seaside_Sparrow
  070.Green_Violetear
  137.Cliff_Swallow
  161.Blue_winged_Warbler
  004.Groove_billed_Ani
  043.Yellow_bellied_Flycatcher
  183.Northern_Waterthrush
  188.Pileated_Woodpecker
  056.Pine_Grosbeak
  122.Harris_Sparrow
  006.Least_Auklet
  085.Horned_Lark
)

dest="clean_top_20"

for cls in "${classes[@]}"; do
	for folder in test_cropped_shuffled train_cropped_augmented test_cropped_shuffled_segmentation train_cropped_augmented_segmentation train_cropped train_cropped_segmentation; do
		echo $cls $folder
		mkdir -p "datasets/cub200_cropped/$dest/$folder"
		cmd="cp -r datasets/cub200_cropped/clean_all/$folder/$cls datasets/cub200_cropped/$dest/$folder"
		$cmd
done
done