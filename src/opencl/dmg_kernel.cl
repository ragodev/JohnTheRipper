/*
 * This software is Copyright (c) 2018 magnum
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 */

#include "pbkdf2_hmac_sha1_kernel.cl"
#define AES_SRC_TYPE MAYBE_CONSTANT
#include "opencl_aes.h"
#include "opencl_des.h"
#include "opencl_sha1_ctx.h"

typedef struct {
	uint dk[((OUTLEN + 19) / 20) * 20 / sizeof(uint)];
	uint cracked;
} dmg_out;

typedef struct {
	uint  length;
	uint  outlen;
	uint  iterations;
	uchar salt[179]; //[243]; is for 4 limb, if we later need it.
	uint ivlen;
	uchar iv[32];
	int headerver;
	uchar chunk[8192];
	uint32_t encrypted_keyblob_size;
	uint8_t encrypted_keyblob[128];
	uint len_wrapped_aes_key;
	uchar wrapped_aes_key[296];
	uint len_hmac_sha1_key;
	uchar wrapped_hmac_sha1_key[300];
	char scp; /* start chunk present */
	uchar zchunk[4096]; /* chunk #0 */
	int cno;
	int data_size;
} dmg_salt;

inline void hmac_sha1(const uchar *_key, uint key_len,
                      const uchar *data, uint data_len,
                      uchar *digest, uint digest_len)
{
	union {
		uchar c[64];
		uint w[64/4];
	} buf;
	uchar local_digest[20];
	uint *pW = (uint*)buf.w;
	SHA_CTX ctx;
	uint i;

#if HMAC_KEY_GT_64
	if (key_len > 64) {
		SHA1_Init(&ctx);
		while (key_len) {
			uchar pbuf[64];
			uint len = MIN(data_len, (uint)sizeof(pbuf));

			memcpy_macro(pbuf, _key, len);
			SHA1_Update(&ctx, pbuf, len);
			data_len -= len;
			_key += len;
		}
		SHA1_Final(buf.c, &ctx);
		pW[0] ^= 0x36363636;
		pW[1] ^= 0x36363636;
		pW[2] ^= 0x36363636;
		pW[3] ^= 0x36363636;
		pW[4] ^= 0x36363636;
		memset_p(&buf.c[20], 0x36, 44);
	} else
#endif
	{
		memcpy_macro(buf.c, _key, key_len);
		memset_p(&buf.c[key_len], 0, 64 - key_len);
		for (i = 0; i < 16; i++)
			pW[i] ^= 0x36363636;
	}
	SHA1_Init(&ctx);
	SHA1_Update(&ctx, buf.c, 64);
	while (data_len) {
		uchar pbuf[64];
		uint len = MIN(data_len, (uint)sizeof(pbuf));

		memcpy_macro(pbuf, data, len);
		SHA1_Update(&ctx, pbuf, len);
		data_len -= len;
		data += len;
	}
	SHA1_Final(local_digest, &ctx);
	for (i = 0; i < 16; i++)
		pW[i] ^= (0x36363636 ^ 0x5c5c5c5c);
	SHA1_Init(&ctx);
	SHA1_Update(&ctx, buf.c, 64);
	SHA1_Update(&ctx, local_digest, 20);
	SHA1_Final(local_digest, &ctx);

	memcpy_pp(digest, local_digest, digest_len);
}

inline int check_pkcs_pad(const uchar* data, int len, int blocksize)
{
	int pad_len = data[len - 1];
	int padding = pad_len;
	int real_len = len - pad_len;
	const uchar *p = data + real_len;

	if (len & (blocksize - 1))
		return -1;

	if (pad_len > blocksize || pad_len < 1)
		return -1;

	if (len < blocksize)
		return -1;

	while (pad_len--)
		if (*p++ != padding)
			return -1;

	return real_len;
}

inline int apple_des3_ede_unwrap_key1(uchar *wrapped_key,
                                      const int wrapped_key_len,
                                      uchar *decryptKey)
{
	des3_context ks;
	uchar TEMP1[sizeof(((dmg_salt*)0)->wrapped_hmac_sha1_key)];
	uchar TEMP2[sizeof(((dmg_salt*)0)->wrapped_hmac_sha1_key)];
	uchar IV[8] = { 0x4a, 0xdd, 0xa2, 0x2c, 0x79, 0xe8, 0x21, 0x05 };
	int outlen, i;

	des3_set3key_dec(&ks, decryptKey);
	des3_crypt_cbc(&ks, DES_DECRYPT, wrapped_key_len, IV, wrapped_key, TEMP1);

	outlen = check_pkcs_pad(TEMP1, wrapped_key_len, 8);
	if (outlen < 0)
		return 0;

	for (i = 0; i < outlen; i++)
		TEMP2[i] = TEMP1[outlen - i - 1];

	outlen -= 8;
	des3_crypt_cbc(&ks, DES_DECRYPT, outlen, TEMP2, TEMP2 + 8, TEMP1);

	outlen = check_pkcs_pad(TEMP1, outlen, 8);
	if (outlen < 0)
		return 0;

	return 1;
}

/* Check for 64-bit NULL at 32-bit alignment */
inline int check_nulls(void *buf, uint size)
{
	uint *p = buf;

	size /= sizeof(size);

	while (--size)
		if (!*p++ && !*p++)
			return 1;
	return 0;
}

inline int hash_plugin_check_hash(uchar *derived_key,
                                  MAYBE_CONSTANT dmg_salt *salt)
{
	dmg_salt private_salt;

	memcpy_mcp(&private_salt, salt, sizeof(dmg_salt));

	if (salt->headerver == 1) {
		if (apple_des3_ede_unwrap_key1(private_salt.wrapped_aes_key, salt->len_wrapped_aes_key, derived_key) &&
		    apple_des3_ede_unwrap_key1(private_salt.wrapped_hmac_sha1_key, salt->len_hmac_sha1_key, derived_key)) {
			return 1;
		}
		return 0;
	}
	else {
		des3_context ks;
		uchar TEMP1[sizeof(salt->wrapped_hmac_sha1_key)];
		AES_KEY aes_decrypt_key;
		uchar outbuf[8192] __attribute__ ((aligned(4)));
		uchar iv[20];
		uchar hmacsha1_key_[20];
		uchar aes_key_[32];

		des3_set3key_dec(&ks, derived_key);
		memcpy_macro(iv, salt->iv, 8);
		des3_crypt_cbc(&ks, DES_DECRYPT, salt->encrypted_keyblob_size,
		               iv, private_salt.encrypted_keyblob, TEMP1);

		memcpy_macro(aes_key_, TEMP1, 32);
		memcpy_macro(hmacsha1_key_, TEMP1, 20);
		hmac_sha1(hmacsha1_key_, 20, (uchar*)&private_salt.cno, 4, iv, 20);
		if (salt->encrypted_keyblob_size == 48)
			AES_set_decrypt_key(aes_key_, 128, &aes_decrypt_key);
		else
			AES_set_decrypt_key(aes_key_, 256, &aes_decrypt_key);
		AES_cbc_decrypt(salt->chunk, outbuf, salt->data_size, &aes_decrypt_key, iv);

		/* 8 consecutive nulls */
		if (check_nulls(outbuf, salt->data_size))
			return 1;

		/* Second buffer test. If present, *this* is the very first block of the DMG */
		if (salt->scp == 1) {
			int cno = 0;

			hmac_sha1(hmacsha1_key_, 20, (uchar*)&cno, 4, iv, 20);
			if (salt->encrypted_keyblob_size == 48)
				AES_set_decrypt_key(aes_key_, 128, &aes_decrypt_key);
			else
				AES_set_decrypt_key(aes_key_, 128 * 2, &aes_decrypt_key);
			AES_cbc_decrypt(salt->zchunk, outbuf, 4096, &aes_decrypt_key, iv);

			/* 8 consecutive nulls */
			if (check_nulls(outbuf, 4096))
				return 1;
		}
	}

	return 0;
}

__kernel
void dmg_final(MAYBE_CONSTANT dmg_salt *salt,
               __global dmg_out *out,
               __global pbkdf2_state *state)
{
	uint dk[OUTLEN / 4];
	uint gid = get_global_id(0);
	uint i;
	uint base = state[gid].pass++ * 5;
	uint pass = state[gid].pass;

	for (i = 0; i < 5; i++)
		out[gid].dk[base + i] = SWAP32(state[gid].out[i]);

	if (4 * base + 20 < OUTLEN) {
		_phsk_hmac_sha1(state[gid].out, state[gid].ipad, state[gid].opad,
		                salt->salt, salt->length, 1 + pass);

		for (i = 0; i < 5; i++)
			state[gid].W[i] = state[gid].out[i];

		state[gid].iter_cnt = salt->iterations - 1;
	} else {
		memcpy_gp(dk, out[gid].dk, OUTLEN);
		out[gid].cracked = hash_plugin_check_hash((uchar*)dk, salt);
	}
}
