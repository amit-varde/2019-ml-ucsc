import pandas as pd
import numpy as np


def aggregate(matches_specs, matches_desc, dataset):
	descset = set(matches_desc)
	specsset = set(matches_specs)
	intersectionset = descset & specsset
	#print("Common Matches between Spec and Desc classifications:")
	#print(intersectionset)
	if len(intersectionset) > 1:
		partnumfilter = dataset['Part Number'].isin(intersectionset)
		filtereddataset = dataset[partnumfilter]
		sorteddata = filtereddataset.sort_values(['Customer'], ascending=False)
		#print("Top Recommendations:")
		#print(sorteddata.head(3))
		return(sorteddata.head(3))
	else:
		print("No common Recommendations:")
		partnumfilterdesc = dataset['Part Number'].isin(descset)
		filtereddatasetdesc = dataset[partnumfilterdesc]
		#print(filtereddatasetdesc)
		partnumfilterspec = dataset['Part Number'].isin(specsset)
		filtereddatasetspec = dataset[partnumfilterspec]
		#print(filtereddatasetspec)
		mergeddataset = pd.merge(filtereddatasetspec, filtereddatasetdesc, how='outer')
		sorteddata = mergeddataset.sort_values(['Customer'], ascending=False)
		#print("Top Recommendations:")
		#print(sorteddata.head(3))
		return(sorteddata.head(3))


def main():
	dataset = pd.read_csv('../data/TI-Opamps.Desc.csv')
	matches_specs = ['OPA4187', 'OPA828', 'OPA2210', 'OPA1671', 'OPA2156', 'OPA462']
	matches_desc = ['OPA4187', 'OPA8281', 'OPA2210', 'OPA1671', 'OPA21567']
	matches_desc_1 = ['TLV9062', 'LM101AQML', 'TLV2221']
	print("-----------------------------------------------")
	print("Aggregation with Intersection:")
	print("-----------------------------------------------")
	aggregate(matches_specs, matches_desc, dataset)
	print("-----------------------------------------------")
	print("Aggregation with No Intersection")
	print("-----------------------------------------------")
	aggregate(matches_specs, matches_desc_1, dataset)


if __name__== "__main__":
    main()
