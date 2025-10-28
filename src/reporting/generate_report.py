"""
generate_report.py
Creates a comprehensive marketing attribution report by combining
narrative summaries with visualizations into markdown, figures, and PDF formats.
"""

import sys
from pathlib import Path
try:
    import bootstrap
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import bootstrap

import markdown
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
from bs4 import BeautifulSoup
from src.utils.logger import logger
from src.reporting.narrative_summary import (
    load_data, 
    analyze_model_performance,
    analyze_attribution,
    analyze_user_behavior,
    generate_markdown_report
)
from src.reporting.visualization import (
    plot_model_performance,
    plot_attribution_analysis,
    plot_user_behavior
)

def convert_markdown_to_html(md_file):
    """Convert markdown to HTML"""
    with open(md_file, 'r', encoding='utf-8') as f:
        md_text = f.read()
    return markdown.markdown(md_text)

def create_pdf_report(html_content, figures_dir, output_pdf):
    """Create PDF with text and figures"""
    logger.info(f"Creating PDF report at {output_pdf}")
    
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Prepare styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='CustomHeading1',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    ))
    styles.add(ParagraphStyle(
        name='CustomHeading2',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=20
    ))
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12
    ))
    
    # Build PDF content
    elements = []
    
    # Add title
    elements.append(Paragraph("Marketing Attribution Analysis Report", styles['CustomHeading1']))
    elements.append(Spacer(1, 12))
    
    # Process each HTML element
    for elem in soup.find_all(['h1', 'h2', 'p', 'ul']):
        if elem.name == 'h1':
            elements.append(Paragraph(elem.text, styles['CustomHeading1']))
        elif elem.name == 'h2':
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(elem.text, styles['CustomHeading2']))
        elif elem.name == 'p':
            elements.append(Paragraph(elem.text, styles['CustomBody']))
        elif elem.name == 'ul':
            for li in elem.find_all('li'):
                elements.append(Paragraph(f"• {li.text}", styles['CustomBody']))
        elements.append(Spacer(1, 12))
    
    # Add figures
    elements.append(Paragraph("Visualizations", styles['CustomHeading1']))
    elements.append(Spacer(1, 12))
    
    figure_files = sorted(figures_dir.glob('*.png'))
    for fig_path in figure_files:
        img = Image(str(fig_path), width=6*inch, height=4*inch, kind='proportional')
        elements.append(img)
        elements.append(Spacer(1, 24))
        elements.append(Paragraph(f"Figure: {fig_path.stem}", styles['CustomBody']))
        elements.append(Spacer(1, 36))
    
    # Build the PDF
    doc.build(elements)

def main():
    """Generate complete attribution analysis report"""
    
    # Set up output directories
    report_dir = bootstrap.OUTPUTS_DIR / "reporting"
    figures_dir = report_dir / "figures"
    full_report_dir = report_dir / "full_report"
    
    # Create all necessary directories
    for dir_path in [report_dir, figures_dir, full_report_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Load and analyze data
    logger.info("Loading and analyzing data...")
    model, df_preds, df_shapley, df_sentiment, df_intent = load_data()
    
    # Generate insights
    model_insights = analyze_model_performance(model, df_preds)
    attribution_insights = analyze_attribution(df_shapley)
    behavior_insights = analyze_user_behavior(df_sentiment, df_intent)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_model_performance(model, df_preds, figures_dir)
    plot_attribution_analysis(df_shapley, figures_dir)
    plot_user_behavior(df_sentiment, df_intent, df_shapley, figures_dir)
    
    # Generate markdown report
    logger.info("Generating markdown report...")
    report_text = generate_markdown_report(model_insights, attribution_insights, behavior_insights)
    
    # Save markdown report
    report_path = report_dir / "marketing_attribution_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    
    logger.info(f"Markdown report saved to {report_path}")
    logger.info(f"Visualizations saved to {figures_dir}")
    
    # Generate PDF report
    logger.info("Generating PDF report...")
    html_content = convert_markdown_to_html(report_path)
    pdf_file = full_report_dir / "marketing_attribution_analysis.pdf"
    create_pdf_report(html_content, figures_dir, pdf_file)
    
    logger.info(f"Complete PDF report saved to {pdf_file}")

if __name__ == "__main__":
    main()