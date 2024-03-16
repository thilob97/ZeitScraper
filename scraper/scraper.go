package scraper

import (
	"ScrapeBot/scraper/constructs"
	"crypto/sha256"
	"encoding/csv"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
	"github.com/gocolly/colly"
	"github.com/gocolly/colly/extensions"
)

type Article = constructs.Article

func Scrape() {
	//just take first 10 for debugging

	count := 0

	fmt.Println("Scraping")
	var articles []Article

	c := colly.NewCollector()
	detailCollector := c.Clone()
	extensions.RandomUserAgent(c)

	detailCollector.OnHTML("article", func(e *colly.HTMLElement) {
		article, err := scrapeDetails(e)
		if err != nil {
			fmt.Println(err)
			return
		}
		articles = append(articles, article)
		count++
	})

	c.OnHTML(".zon-teaser__link", func(e *colly.HTMLElement) {
		if count > 99999999 { //just take first 10 for debugging
			return
		}
		article_link := foundArticle(e)
		detailCollector.Visit(article_link)
	})

	c.Visit("https://www.zeit.de/news/index")

	writeCSV(articles)
}

func foundArticle(e *colly.HTMLElement) string {
	return e.Attr("href")
}

func scrapeDetails(e *colly.HTMLElement) (Article, error) {
	Title := e.ChildText(".article-heading__title")

	if Title == "" {
		return Article{}, fmt.Errorf("No title found")
	}

	var keywords, Published, LastUpdated, Description string

	e.DOM.ParentsUntil("~").Find("meta[name='keywords']").Each(func(_ int, s *goquery.Selection) {
		content, exists := s.Attr("content")
		if exists {
			keywords = content
		}
	})

	e.DOM.ParentsUntil("~").Find("meta[name='date']").Each(func(_ int, s *goquery.Selection) {
		content, exists := s.Attr("content")
		if exists {
			Published = content
		}
	})

	e.DOM.ParentsUntil("~").Find("meta[name='last-modified']").Each(func(_ int, s *goquery.Selection) {
		content, exists := s.Attr("content")
		if exists {
			LastUpdated = content
		}
	})

	e.DOM.ParentsUntil("~").Find("meta[name='description']").Each(func(_ int, s *goquery.Selection) {
		content, exists := s.Attr("content")
		if exists {
			Description = content
		}
	})

	source := strings.TrimPrefix(e.ChildText(".metadata__source"), "Quelle: ")
	source = strings.ReplaceAll(source, "\n", "")
	source = strings.TrimSpace(source)

	var paywall bool
	if e.ChildText(".zplus-badge__text") != "" {
		paywall = true
	} else {
		paywall = false
	}

	ArticleTitle := e.ChildText(".article-heading__title")

	MD5Title := fmt.Sprintf("%x", sha256.Sum256([]byte(ArticleTitle)))

	return Article{
		Hash:        MD5Title,
		Title:       ArticleTitle,
		Link:        e.Request.URL.String(),
		Description: Description,
		Category:    e.ChildText(".article-heading__kicker"),
		Author:      e.ChildText("[itemprop='name']"),
		KeyWords:    keywords,
		Published:   Published,
		LastUpdated: LastUpdated,
		Source:      source,
		Paywall:     paywall,
	}, nil
}

func writeCSV(articles []Article) {
	err := os.MkdirAll("./data/raw/articles", os.ModePerm)
	if err != nil {
		fmt.Println(err)
	}

	currentTime := time.Now()

	file, err := os.Create("./data/raw/articles/article_" + currentTime.Format("20060102_150405") + ".csv")
	if err != nil {
		fmt.Println(err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	writer.Comma = ';'

	file.WriteString("Hash;Title;Link;Description;Category;KeyWords;Published;LastUpdated;Author;Source;Paywall\n")

	for _, article := range articles {
		err := writer.Write([]string{
			article.Hash,
			article.Title,
			article.Link,
			article.Description,
			article.Category,
			article.KeyWords,
			article.Published,
			article.LastUpdated,
			article.Author,
			article.Source,
			fmt.Sprintf("%t", article.Paywall),
		})
		if err != nil {
			fmt.Println(err)
		}
	}

	fmt.Println("File written successfully")
}
