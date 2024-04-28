---
layout: post
title:  让Google检索到你的博文
date:   2024-04-28 14:53:59 +0800
tags:   随笔 seo
---

从在Github Pages写博客起，就从未关心过这些文章能否被搜索引擎找到。昨天翻了翻博客，发现有些文章写得挺详细，与其自娱自乐，不如大家共享，于是准备让谷歌检索这些文章。

1. 进入[Google Search Console](https://search.google.com/search-console/welcome)，输入个人主页的地址，例如https://xxxx.github.io
    <p align="center"><img src="/assets/img/let-google-search-your-pages/google-search-console-welcome.PNG" alt="google-search-console-welcome" width="100%"></p>

2. 验证所有权。验证方法有很多，例如在首页添加元标记等，但是推荐的方法是下载Google生成的html验证文件，然后上传到个人网站
    <p align="center"><img src="/assets/img/let-google-search-your-pages/verify-site.PNG" alt="verify-site" width="100%"></p>

3. 生成站点地图。以我的[个人主页](https://snailcoder.github.io)为例，进入[XLM sitemaps generator](https://www.xml-sitemaps.com/)，输入主页地址就能得到站点地图，然后下载文件sitemap.xml
    <p align="center"><img src="/assets/img/let-google-search-your-pages/xml-sitemap-generator.PNG" alt="xml-sitemap-generator" width="100%"></p>

4. 提交站点地图。把sitemap.xml上传到个人网站，然后在Google Search Console中添加站点地图，让Google为这个站点编制索引
    <p align="center"><img src="/assets/img/let-google-search-your-pages/add-sitemap.PNG" alt="add-sitemap" width="100%"></p>

5. 至此，个人主页已经被Google“盯上”了，一段时间之后Google就能检索到个人主页了。具体时长可能是几天或几周，可以参考[这里的讨论](https://www.quora.com/How-long-does-it-usually-take-for-a-website-to-show-up-in-Google-after-the-sitemap-is-submitted-and-indexing-is-requested)。但是，还有一个方法可以让Google更快检索到你的主页：使用Google Search Console中的***网址检查***功能。例如，我想让[这篇文章](https://snailcoder.github.io/2024/03/31/parameter-efficient-llm-fine-tuning-soft-prompt.html)更快被检索到，就点击“网址检查”，输入这篇文章的地址：
    <p align="center"><img src="/assets/img/let-google-search-your-pages/check-site.PNG" alt="check-site" width="100%"></p>
    得到的结果是“网址尚未收录到Google”，这时点击***请求编入索引***，你的请求会被放到一个优先级队列中，稍等片刻即可成功：
    <p align="center"><img src="/assets/img/let-google-search-your-pages/already-indexed.PNG" alt="already-indexed" width="100%"></p>

上面几个步骤做完，用不了多久能在Google搜到自己的博客了。

