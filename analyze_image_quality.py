#!/usr/bin/env python3
"""
Standalone Image Quality Analysis Script
Analyzes image quality for all images in a directory using the ReceiptDamageAnalyzer.
"""

import argparse
import json
import os
import pathlib
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

from damage_detection import ReceiptDamageAnalyzer
from agno.utils.log import logger


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class ImageQualityBatchProcessor:
    """Batch processor for analyzing image quality in directories."""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.analyzer = ReceiptDamageAnalyzer(debug_mode=debug_mode)
        self.supported_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'}
        
    def find_image_files(self, input_dir: pathlib.Path) -> List[pathlib.Path]:
        """Find all supported image files in the directory."""
        image_files = []
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
        
        # Recursively find all image files
        for file_path in input_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                image_files.append(file_path)
        
        return sorted(image_files)
    
    def analyze_single_image(self, image_path: pathlib.Path) -> Dict:
        """Analyze quality of a single image file."""
        try:
            # Use the same logic as the Streamlit version
            quality_results = self.analyzer.analyze_receipt_damage(str(image_path))
            
            # Add metadata
            quality_results['metadata'] = {
                'file_path': str(image_path),
                'file_name': image_path.name,
                'file_size': image_path.stat().st_size,
                'analysis_timestamp': datetime.now().isoformat(),
                'file_extension': image_path.suffix.lower()
            }
            
            return quality_results
            
        except Exception as e:
            logger.error(f"Image quality analysis failed for {image_path}: {str(e)}")
            return {
                "error": str(e),
                "overall_score": 0.0,
                "ocr_suitable": {"suitable": False, "confidence": "error", "expected_accuracy": "unknown"},
                "damage_details": {
                    "folds": {"coverage": 0, "severity": "unknown"},
                    "tears": {"coverage": 0, "severity": "unknown"},
                    "stains": {"coverage": 0, "severity": "unknown"},
                    "contrast": {"quality": "unknown"}
                },
                'metadata': {
                    'file_path': str(image_path),
                    'file_name': image_path.name,
                    'file_size': image_path.stat().st_size if image_path.exists() else 0,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'file_extension': image_path.suffix.lower(),
                    'error': True
                }
            }
    
    def process_directory(self, input_dir: pathlib.Path, output_dir: pathlib.Path) -> Tuple[List[Dict], Dict]:
        """Process all images in a directory and return results."""
        # Find all image files
        image_files = self.find_image_files(input_dir)
        
        if not image_files:
            print(f"No supported image files found in {input_dir}")
            print(f"Supported extensions: {', '.join(self.supported_extensions)}")
            return [], {}
        
        print(f"Found {len(image_files)} image files to analyze")
        print(f"Supported extensions: {', '.join(self.supported_extensions)}")
        print("-" * 60)
        
        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        individual_results_dir = output_dir / "individual_results"
        individual_results_dir.mkdir(exist_ok=True)
        
        # Process each image
        results = []
        processing_stats = {
            'total_files': len(image_files),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'start_time': datetime.now().isoformat(),
            'processing_times': []
        }
        
        for i, image_path in enumerate(image_files, 1):
            start_time = time.time()
            
            print(f"[{i:3d}/{len(image_files)}] Analyzing: {image_path.name}")
            
            # Analyze image quality
            quality_result = self.analyze_single_image(image_path)
            results.append(quality_result)
            
            # Track processing time
            processing_time = time.time() - start_time
            processing_stats['processing_times'].append(processing_time)
            
            # Update stats
            if 'error' in quality_result:
                processing_stats['failed_analyses'] += 1
                print(f"    ❌ Failed: {quality_result['error']}")
            else:
                processing_stats['successful_analyses'] += 1
                score = quality_result['overall_score']
                ocr_suitable = quality_result['ocr_suitable']['suitable']
                confidence = quality_result['ocr_suitable']['confidence']
                print(f"    ✅ Score: {score:.3f} | OCR: {'✓' if ocr_suitable else '✗'} ({confidence})")
            
            # Save individual result
            result_filename = f"{image_path.stem}_quality.json"
            result_path = individual_results_dir / result_filename
            
            try:
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(quality_result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            except Exception as e:
                print(f"    ⚠️  Warning: Could not save result file: {e}")
            
            # Show progress
            progress = (i / len(image_files)) * 100
            print(f"    Progress: {progress:.1f}% | Time: {processing_time:.2f}s")
            print()
        
        # Finalize processing stats
        processing_stats['end_time'] = datetime.now().isoformat()
        processing_stats['total_processing_time'] = sum(processing_stats['processing_times'])
        processing_stats['average_processing_time'] = (
            processing_stats['total_processing_time'] / len(processing_stats['processing_times'])
            if processing_stats['processing_times'] else 0
        )
        
        return results, processing_stats
    
    def generate_summary_report(self, results: List[Dict], processing_stats: Dict) -> Dict:
        """Generate a summary report of all analyses."""
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        summary = {
            'processing_summary': processing_stats,
            'analysis_summary': {
                'total_files_analyzed': len(results),
                'successful_analyses': len(successful_results),
                'failed_analyses': len(failed_results),
                'success_rate': len(successful_results) / len(results) if results else 0
            }
        }
        
        if successful_results:
            # Calculate statistics for successful analyses
            scores = [r['overall_score'] for r in successful_results]
            ocr_suitable_count = sum(1 for r in successful_results if r['ocr_suitable']['suitable'])
            
            # Quality distribution
            excellent_count = sum(1 for s in scores if s >= 0.8)
            good_count = sum(1 for s in scores if 0.6 <= s < 0.8)
            fair_count = sum(1 for s in scores if 0.4 <= s < 0.6)
            poor_count = sum(1 for s in scores if s < 0.4)
            
            # Damage statistics
            fold_coverages = [r['damage_details']['folds']['coverage'] for r in successful_results]
            tear_coverages = [r['damage_details']['tears']['coverage'] for r in successful_results]
            stain_coverages = [r['damage_details']['stains']['coverage'] for r in successful_results]
            
            summary['quality_statistics'] = {
                'overall_scores': {
                    'min': min(scores),
                    'max': max(scores),
                    'average': sum(scores) / len(scores),
                    'median': sorted(scores)[len(scores) // 2]
                },
                'quality_distribution': {
                    'excellent (≥0.8)': excellent_count,
                    'good (0.6-0.8)': good_count,
                    'fair (0.4-0.6)': fair_count,
                    'poor (<0.4)': poor_count
                },
                'ocr_suitability': {
                    'suitable_count': ocr_suitable_count,
                    'unsuitable_count': len(successful_results) - ocr_suitable_count,
                    'suitability_rate': ocr_suitable_count / len(successful_results)
                },
                'damage_statistics': {
                    'average_fold_coverage': sum(fold_coverages) / len(fold_coverages),
                    'average_tear_coverage': sum(tear_coverages) / len(tear_coverages),
                    'average_stain_coverage': sum(stain_coverages) / len(stain_coverages),
                    'max_fold_coverage': max(fold_coverages),
                    'max_tear_coverage': max(tear_coverages),
                    'max_stain_coverage': max(stain_coverages)
                }
            }
        
        if failed_results:
            # Error analysis
            error_types = {}
            for result in failed_results:
                error = result.get('error', 'Unknown error')
                error_types[error] = error_types.get(error, 0) + 1
            
            summary['error_analysis'] = {
                'error_types': error_types,
                'most_common_error': max(error_types.items(), key=lambda x: x[1]) if error_types else None
            }
        
        return summary
    
    def save_results(self, results: List[Dict], processing_stats: Dict, output_dir: pathlib.Path):
        """Save all results and generate reports."""
        # Generate summary report
        summary_report = self.generate_summary_report(results, processing_stats)
        
        # Save summary report
        summary_path = output_dir / "summary_report.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        # Save complete results
        complete_results_path = output_dir / "complete_results.json"
        with open(complete_results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'results': results,
                'summary': summary_report
            }, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        # Generate processing log
        log_path = output_dir / "processing_log.txt"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("IMAGE QUALITY ANALYSIS LOG\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files processed: {processing_stats['total_files']}\n")
            f.write(f"Successful analyses: {processing_stats['successful_analyses']}\n")
            f.write(f"Failed analyses: {processing_stats['failed_analyses']}\n")
            f.write(f"Success rate: {(processing_stats['successful_analyses']/processing_stats['total_files']*100):.1f}%\n")
            f.write(f"Total processing time: {processing_stats['total_processing_time']:.2f} seconds\n")
            f.write(f"Average time per file: {processing_stats['average_processing_time']:.2f} seconds\n\n")
            
            if 'quality_statistics' in summary_report:
                stats = summary_report['quality_statistics']
                f.write("QUALITY STATISTICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average quality score: {stats['overall_scores']['average']:.3f}\n")
                f.write(f"Score range: {stats['overall_scores']['min']:.3f} - {stats['overall_scores']['max']:.3f}\n")
                f.write(f"OCR suitability rate: {stats['ocr_suitability']['suitability_rate']*100:.1f}%\n\n")
                
                f.write("QUALITY DISTRIBUTION\n")
                f.write("-" * 20 + "\n")
                for category, count in stats['quality_distribution'].items():
                    f.write(f"{category}: {count} files\n")
        
        return summary_path, complete_results_path, log_path


def print_summary_stats(summary_report: Dict):
    """Print a formatted summary of the analysis results."""
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    analysis_summary = summary_report['analysis_summary']
    print(f"Total files analyzed: {analysis_summary['total_files_analyzed']}")
    print(f"Successful analyses: {analysis_summary['successful_analyses']}")
    print(f"Failed analyses: {analysis_summary['failed_analyses']}")
    print(f"Success rate: {analysis_summary['success_rate']*100:.1f}%")
    
    if 'quality_statistics' in summary_report:
        stats = summary_report['quality_statistics']
        print(f"\nQUALITY STATISTICS")
        print("-" * 30)
        print(f"Average quality score: {stats['overall_scores']['average']:.3f}")
        print(f"Score range: {stats['overall_scores']['min']:.3f} - {stats['overall_scores']['max']:.3f}")
        print(f"OCR suitability rate: {stats['ocr_suitability']['suitability_rate']*100:.1f}%")
        
        print(f"\nQUALITY DISTRIBUTION")
        print("-" * 30)
        for category, count in stats['quality_distribution'].items():
            print(f"{category}: {count} files")
    
    processing_stats = summary_report['processing_summary']
    print(f"\nPROCESSING PERFORMANCE")
    print("-" * 30)
    print(f"Total processing time: {processing_stats['total_processing_time']:.2f} seconds")
    print(f"Average time per file: {processing_stats['average_processing_time']:.2f} seconds")


def main():
    """Main function to run the image quality analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze image quality for all images in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_image_quality.py --input-dir ./expense_files
  python analyze_image_quality.py --input-dir ./images --output-dir ./results --debug
  python analyze_image_quality.py -i ./photos -o ./analysis_results --no-summary
        """
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        required=True,
        help='Input directory containing images to analyze'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='image_quality_results',
        help='Output directory for results (default: image_quality_results)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with visualizations'
    )
    
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip printing summary statistics to console'
    )
    
    args = parser.parse_args()
    
    # Convert paths to pathlib objects
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    
    try:
        # Initialize processor
        processor = ImageQualityBatchProcessor(debug_mode=args.debug)
        
        print("IMAGE QUALITY ANALYSIS")
        print("=" * 60)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
        print()
        
        # Process directory
        results, processing_stats = processor.process_directory(input_dir, output_dir)
        
        if not results:
            print("No images were processed. Exiting.")
            sys.exit(1)
        
        # Save results
        print("Saving results...")
        summary_path, complete_path, log_path = processor.save_results(results, processing_stats, output_dir)
        
        print(f"✅ Results saved to:")
        print(f"   Summary report: {summary_path}")
        print(f"   Complete results: {complete_path}")
        print(f"   Processing log: {log_path}")
        print(f"   Individual results: {output_dir / 'individual_results'}")
        
        # Print summary statistics
        if not args.no_summary:
            summary_report = processor.generate_summary_report(results, processing_stats)
            print_summary_stats(summary_report)
        
        print(f"\n🎉 Analysis complete! Processed {len(results)} images.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
