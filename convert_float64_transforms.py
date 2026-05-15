#!/usr/bin/env python3

import argparse
import csv
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Convert BIDS subject HDF5 transforms ending in _xfm.h5 to single '
            'precision using antsApplyTransforms, with per-transform logging.'
            'Requires antsApplyTransforms (>= 2.6.3) to be on the PATH.'
        )
    )
    parser.add_argument(
        'bids_dataset',
        help='Path to the BIDS dataset directory',
    )
    parser.add_argument(
        'log_file',
        help='Path to a CSV file containing logs of files updated',
    )
    parser.add_argument(
        'subject_list',
        help='Path to a text file containing one subject ID per line (without sub- prefix)',
    )
    return parser


def compute_md5(path):
    md5 = hashlib.md5()
    with open(path, 'rb') as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b''):
            md5.update(chunk)
    return md5.hexdigest()


def log_row(stdout_writer, rows, subject, transform_file, success, error):
    row = {
        'subject': subject,
        'transform_file': transform_file,
        'success': success,
        'error': error,
    }
    rows.append(row)
    stdout_writer.writerow([subject, transform_file, success, error])
    sys.stdout.flush()


def read_subjects(subject_list_path):
    subjects = []
    with open(subject_list_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            subject = line.strip()
            if subject:
                subjects.append(subject)
    return subjects


def find_transforms(subject_dir):
    return sorted(
        path for path in subject_dir.rglob('*_xfm.h5')
        if path.is_file()
    )


def check_ants_apply_transforms_available():
    try:
        subprocess.run(
            ['antsApplyTransforms', '--version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        print(
            'Error: antsApplyTransforms was not found. '
            'Please ensure antsApplyTransforms is on the PATH.',
            file=sys.stderr,
        )
        sys.exit(1)


def is_effectively_writable(path):
    parent = path.parent
    return os.access(path, os.W_OK) and os.access(parent, os.W_OK)


def get_file_mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def restore_backup_if_needed(transform_path, backup_path, original_mode=None):
    if backup_path.exists():
        if transform_path.exists():
            transform_path.unlink()
        backup_path.replace(transform_path)
        if original_mode is not None:
            os.chmod(transform_path, original_mode)


def process_transform(dataset_dir, subject, transform_path, stdout_writer, rows):
    relpath = str(transform_path.relative_to(dataset_dir))
    backup_path = transform_path.parent / f"float_convert_backup_{transform_path.name}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tempfloat_path = tmpdir_path / transform_path.name
        backup_created = False
        original_mode = None

        try:
            if not is_effectively_writable(transform_path):
                log_row(
                    stdout_writer,
                    rows,
                    subject,
                    relpath,
                    False,
                    'transform file or parent directory is not writable',
                )
                return

            if backup_path.exists():
                log_row(
                    stdout_writer,
                    rows,
                    subject,
                    relpath,
                    False,
                    f"backup file already exists: {backup_path.name}",
                )
                return

            original_mode = get_file_mode(transform_path)

            command = [
                'antsApplyTransforms',
                '-d',
                '3',
                '-t',
                str(transform_path),
                '-o',
                f'CompositeTransform[{tempfloat_path}]',
                '--float',
                '--verbose',
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )

            if result.returncode != 0:
                log_row(
                    stdout_writer,
                    rows,
                    subject,
                    relpath,
                    False,
                    f"antsApplyTransforms exited with code {result.returncode}",
                )
                print(result.stdout, file=sys.stderr)
                return

            if not tempfloat_path.exists():
                log_row(
                    stdout_writer,
                    rows,
                    subject,
                    relpath,
                    False,
                    f"antsApplyTransforms did not create output file: {tempfloat_path}",
                )
                return

            tempfloat_md5 = compute_md5(tempfloat_path)

            transform_path.replace(backup_path)
            backup_created = True

            shutil.copy2(tempfloat_path, transform_path)
            os.chmod(transform_path, original_mode)

            new_transform_md5 = compute_md5(transform_path)
            if new_transform_md5 != tempfloat_md5:
                restore_backup_if_needed(transform_path, backup_path, original_mode)
                log_row(
                    stdout_writer,
                    rows,
                    subject,
                    relpath,
                    False,
                    (
                        'md5 mismatch after copying converted transform; '
                        f'original restored from backup {backup_path.name}; '
                        f'temporary converted file was {tempfloat_path}'
                    ),
                )
                return

            backup_path.unlink()
            backup_created = False

            log_row(
                stdout_writer,
                rows,
                subject,
                relpath,
                True,
                'NA',
            )

        except Exception as exc:
            try:
                if backup_created:
                    restore_backup_if_needed(transform_path, backup_path, original_mode)
            except Exception as restore_exc:
                error_message = (
                    f"exception during conversion: {exc}; "
                    f"additionally failed to restore original from backup: {restore_exc}"
                )
            else:
                error_message = f"exception during conversion: {exc}"

            log_row(
                stdout_writer,
                rows,
                subject,
                relpath,
                False,
                error_message,
            )


def main():
    parser = build_arg_parser()

    if len(sys.argv) == 1:
        parser.print_usage(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    check_ants_apply_transforms_available()

    dataset_dir = Path(args.bids_dataset).resolve()
    subject_list_path = Path(args.subject_list).resolve()

    stdout_writer = csv.writer(
        sys.stdout,
        lineterminator='\n',
        quoting=csv.QUOTE_MINIMAL,
    )
    stdout_writer.writerow(['subject', 'transform_file', 'success', 'error'])
    sys.stdout.flush()

    rows = []

    try:
        subjects = read_subjects(subject_list_path)
    except Exception as exc:
        df = pd.DataFrame(rows, columns=['subject', 'transform_file', 'success', 'error'])
        df.to_csv('transform_conversion_log.csv', index=False)
        raise RuntimeError(f'failed to read subject list: {exc}') from exc

    for subject in subjects:
        subject_dir = dataset_dir / f'sub-{subject}'
        if not subject_dir.is_dir():
            continue

        try:
            transforms = find_transforms(subject_dir)
        except Exception as exc:
            log_row(
                stdout_writer,
                rows,
                subject,
                '',
                False,
                f'failed to search for transforms: {exc}',
            )
            continue

        if not transforms:
            continue

        for transform_path in transforms:
            process_transform(dataset_dir, subject, transform_path, stdout_writer, rows)

    df = pd.DataFrame(rows, columns=['subject', 'transform_file', 'success', 'error'])
    df.to_csv('transform_conversion_log.csv', index=False)


if __name__ == '__main__':
    main()
