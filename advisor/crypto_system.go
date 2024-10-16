package main

import (
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
)

type CryptoSystem struct {
	encoder   *hefloat.Encoder
	encryptor *rlwe.Encryptor
	params    hefloat.Parameters
	evaluator *hefloat.Evaluator
	evl       *bootstrapping.Evaluator
}

func InitCryptoSystem() CryptoSystem {
	LogN := 16
	//max_key_length := 20

	params, err := hefloat.NewParametersFromLiteral(
		hefloat.ParametersLiteral{
			LogN:            LogN,
			LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
			LogP:            []int{61, 61, 61},
			LogDefaultScale: 40,
		})
	if err != nil {
		panic(err)
	}

	btpParametersLit := bootstrapping.ParametersLiteral{
		LogP: []int{61, 61, 61, 61},
		Xs:   params.Xs(),
	}
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, btpParametersLit)
	if err != nil {
		panic(err)
	}

	kgen := hefloat.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()

	encoder := hefloat.NewEncoder(params)
	encryptor := hefloat.NewEncryptor(params, pk)

	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	eval := hefloat.NewEvaluator(params, evk)
	evak, _, _ := btpParams.GenEvaluationKeys(sk)

	var evl *bootstrapping.Evaluator
	if evl, err = bootstrapping.NewEvaluator(btpParams, evak); err != nil {
		panic(err)
	}

	galEls := []uint64{
		params.GaloisElement(1),
		params.GaloisElement(2),
		params.GaloisElement(4),
		params.GaloisElement(8),
		params.GaloisElement(16),
		params.GaloisElement(32),
		params.GaloisElement(64),
		params.GaloisElement(128),
		params.GaloisElement(256),
		params.GaloisElement(512),
		params.GaloisElement(1024),
		params.GaloisElement(2048),
		params.GaloisElement(4096),
		params.GaloisElement(8192),
	}
	eval = eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galEls, sk)...))

	return CryptoSystem{
		encoder:   encoder,
		encryptor: encryptor,
		params:    params,
		evaluator: eval,
		evl:       evl,
	}

}

func generateCiphertexts(batch_size int, cryptoSystem CryptoSystem) []*rlwe.Ciphertext {
	var vect1 []complex128
	var cts []*rlwe.Ciphertext
	for i := 0; i < batch_size; i++ {
		pt := hefloat.NewPlaintext(cryptoSystem.params, cryptoSystem.params.MaxLevel())
		if err := cryptoSystem.encoder.Encode(vect1, pt); err != nil {
			panic(err)
		}
		ct, _ := cryptoSystem.encryptor.EncryptNew(pt)
		cts = append(cts, ct)
	}
	return cts
}
