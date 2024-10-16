package main

type NetworkArchitecture struct {
	Layers   []int // Full sequence of layer sizes
	SplitIdx int   // Index in Layers where the split occurs (index 'c')
}

func NewNetworkArchitecture(layers []int, splitIdx int) *NetworkArchitecture {
	return &NetworkArchitecture{
		Layers:   layers,
		SplitIdx: splitIdx,
	}
}
