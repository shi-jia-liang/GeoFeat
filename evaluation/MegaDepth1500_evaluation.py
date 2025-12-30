        'num_pairs_failed': failed_pairs,
        'auc': {k: float(v) for k, v in d_err_auc.items()},
        'maa': {
            f'maa@{t}': float((errors <= t).sum() / len(errors))
            for t in [5, 10, 20]
        },
        'errors': errors.tolist()
    }
    
    with open(result_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"[OK] 结果已保存到: {result_file}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
